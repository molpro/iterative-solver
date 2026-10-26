#include "util.h"

#include <charconv>
#include <stdexcept>
#include <system_error>

namespace molpro::linalg::itsolv::util {

std::string StringFacet::toupper(std::string in) const {
  facet.toupper(in.data(), in.data() + in.size());
  return in;
}

std::string StringFacet::tolower(std::string in) const {
  facet.tolower(in.data(), in.data() + in.size());
  return in;
}

bool StringFacet::tobool(const std::string &in) const {
  auto val = toupper(in);
  crop_space(val);
  bool result;
  if (val == "TRUE" || val == "T" || val == "1") {
    result = true;
  } else if (val == "FALSE" || val == "F" || val == "0") {
    result = false;
  } else {
    throw std::runtime_error("value =" + val + ", must be one of {TRUE, T, 1, FALSE, F, 0}");
  }
  return result;
}

void StringFacet::crop_space(std::string &path) {
  path.erase(path.begin(), std::find_if(path.begin(), path.end(), [](auto &el) { return !std::isspace(el); }));
  path.erase(std::find_if(path.rbegin(), path.rend(), [](auto &el) { return !std::isspace(el); }).base(), path.end());
}

static std::string trim_string(std::string s) {
  StringFacet::crop_space(s);
  return s;
}

std::size_t tosize(const std::string &key, const std::string &value) {
  const auto s = trim_string(value);
  std::size_t result = 0;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), result);
  if (ec == std::errc::result_out_of_range)
    throw std::invalid_argument("option " + key + " = " + value + " is too large");
  if (s.empty() || ec != std::errc{} || end != s.data() + s.size())
    throw std::invalid_argument("option " + key + " = " + value + " is not a non-negative integer");
  return result;
}

std::map<std::string, std::string> StringFacet::parse_keyval_string(std::string s) {
  std::map<std::string, std::string> result;
  const std::string field_separators = ",;";
  const std::string keyval_separators = "=:";
  s += field_separators;
  crop_space(s);
  while (!s.empty()) {
    auto end = s.find_first_of(field_separators);
    auto entry = trim_string(s.substr(0, end));
    if (entry.empty())
      break;
    auto equal = entry.find_first_of(keyval_separators);
    if (equal == std::string::npos)
      throw std::runtime_error("String " + entry + " cannot be parsed as key" + field_separators.front() + "value");
    result[trim_string(entry.substr(0, equal))] = trim_string(entry.substr(equal + 1));
    s = trim_string(s.substr(end + 1));
  }
  return result;
}

} // namespace molpro::linalg::itsolv::util
