<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.4">
  <compound kind="struct">
    <name>molpro::linalg::array::array_family</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1array__family.html</filename>
    <templarg>class T</templarg>
    <templarg>bool</templarg>
    <templarg>bool</templarg>
    <templarg>bool</templarg>
    <templarg>bool</templarg>
    <member kind="function">
      <type>constexpr auto</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1array__family.html</anchorfile>
      <anchor>a5a3844cc871b128e4b953e526314aae7</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::array_family&lt; T, false, false, true, false &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1array__family_3_01T_00_01false_00_01false_00_01true_00_01false_01_4.html</filename>
    <templarg>class T</templarg>
    <member kind="function">
      <type>constexpr auto</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1array__family_3_01T_00_01false_00_01false_00_01true_00_01false_01_4.html</anchorfile>
      <anchor>a047177d42a58f7841a2f15cd468e862c</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::array_family&lt; T, false, false, true, true &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1array__family_3_01T_00_01false_00_01false_00_01true_00_01true_01_4.html</filename>
    <templarg>class T</templarg>
    <member kind="function">
      <type>constexpr auto</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1array__family_3_01T_00_01false_00_01false_00_01true_00_01true_01_4.html</anchorfile>
      <anchor>aa86fda6a11ed1d975bee6489e483f517</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::array_family&lt; T, false, true, false, false &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1array__family_3_01T_00_01false_00_01true_00_01false_00_01false_01_4.html</filename>
    <templarg>class T</templarg>
    <member kind="function">
      <type>constexpr auto</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1array__family_3_01T_00_01false_00_01true_00_01false_00_01false_01_4.html</anchorfile>
      <anchor>addb96de1f6a5f0011ecc1e87371b22a3</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::array_family&lt; T, true, false, false, false &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1array__family_3_01T_00_01true_00_01false_00_01false_00_01false_01_4.html</filename>
    <templarg>class T</templarg>
    <member kind="function">
      <type>constexpr auto</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1array__family_3_01T_00_01true_00_01false_00_01false_00_01false_01_4.html</anchorfile>
      <anchor>afcc8f7e56c336265087110bb90388bbf</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandler</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</filename>
    <templarg>class AL</templarg>
    <templarg>class AR</templarg>
    <class kind="struct">molpro::linalg::array::ArrayHandler::Counter</class>
    <class kind="class">molpro::linalg::array::ArrayHandler::LazyHandle</class>
    <class kind="class">molpro::linalg::array::ArrayHandler::ProxyHandle</class>
    <member kind="typedef">
      <type>typename array::mapped_or_value_type_t&lt; AL &gt;</type>
      <name>value_type_L</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a421d6107b2a60ffcdfb79d2d27c30c73</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::mapped_or_value_type_t&lt; AR &gt;</type>
      <name>value_type_R</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a1750335b1260644a4dd836929ba1e2bf</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>decltype(value_type_L{} *value_type_R{})</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a439d2213a07466b54e366f4fa45b02de</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>decltype(check_abs&lt; value_type &gt;())</type>
      <name>value_type_abs</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>abb16bb06ec4dfd2118e55f64087a204e</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a4e3e408ae09aaf1a463257dec668cb1f</anchor>
      <arglist>(const AR &amp;source)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a6788b6924c2e9565d4cbac8e6c84109f</anchor>
      <arglist>(AL &amp;x, const AR &amp;y)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a2d055c67fa2b7087b03ada96877b842c</anchor>
      <arglist>(value_type alpha, AL &amp;x)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a8c8021df8b9b0c3c449781783eb8a242</anchor>
      <arglist>(value_type alpha, AL &amp;x)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a6e145dd6a76d8ddea588f2104a9c140e</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a05ed79049a10cf88164895aa2da8c6c6</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a425ca68c78689a5eba39acdba92f364c</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a8882095af0c40a0f6c678fec6f051391</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a961a362008722b7118640c95eb5189ac</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::map&lt; size_t, value_type &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a0cd8a03ec5ff22776cb80dacf55c0552</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false)=0</arglist>
    </member>
    <member kind="function">
      <type>const Counter &amp;</type>
      <name>counter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>abe05067f4eb67264edaef960015a9423</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>counter_to_string</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>aaeef8109bd1d949e8f413a399c0c29be</anchor>
      <arglist>(std::string L, std::string R)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>clear_counter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>af6e2096242aea7daef3d8f0c1ee8df06</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~ArrayHandler</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a9adcb3b8f3875d8706e792a0e09d5780</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a68c07505bdf99acceea83ad5d326859f</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>ArrayHandler</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>aab00b2597df72f3989554c25f8a55bdb</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>ArrayHandler</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a5e709cde75bb14f82ddddfb63d7d4e7e</anchor>
      <arglist>(const ArrayHandler &amp;)=default</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual void</type>
      <name>error</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>ad70d4a3f485411537313606452fb75e6</anchor>
      <arglist>(const std::string &amp;message)</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual void</type>
      <name>fused_axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a8d3ebad34f0f0b94a12816e50c7f3ee6</anchor>
      <arglist>(const std::vector&lt; std::tuple&lt; size_t, size_t, size_t &gt; &gt; &amp;reg, const std::vector&lt; value_type &gt; &amp;alphas, const std::vector&lt; std::reference_wrapper&lt; const AR &gt; &gt; &amp;xx, std::vector&lt; std::reference_wrapper&lt; AL &gt; &gt; &amp;yy)</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual void</type>
      <name>fused_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a0cd5e391cf70fa6cf09acc8f72a55830</anchor>
      <arglist>(const std::vector&lt; std::tuple&lt; size_t, size_t, size_t &gt; &gt; &amp;reg, const std::vector&lt; std::reference_wrapper&lt; const AL &gt; &gt; &amp;xx, const std::vector&lt; std::reference_wrapper&lt; const AR &gt; &gt; &amp;yy, std::vector&lt; std::reference_wrapper&lt; value_type &gt; &gt; &amp;out)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>save_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>adffb2fe5e20a200af2a6656138d79907</anchor>
      <arglist>(const std::shared_ptr&lt; LazyHandle &gt; &amp;handle)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>acb672cde1dc9b5ed521c8efe0ae1b18e</anchor>
      <arglist>(ArrayHandler&lt; AL, AR &gt; &amp;handler)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::unique_ptr&lt; Counter &gt;</type>
      <name>m_counter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a07fc7fcd247c2bc725a97ac15f2f7cb2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; std::weak_ptr&lt; LazyHandle &gt; &gt;</type>
      <name>m_lazy_handles</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler.html</anchorfile>
      <anchor>a261ade4cb7ffb330d8876acda683440d</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDDisk</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</filename>
    <templarg>class AL</templarg>
    <templarg>class AR</templarg>
    <base>ArrayHandler&lt; AL, AL &gt;</base>
    <member kind="function">
      <type></type>
      <name>ArrayHandlerDDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a1f01a56394550e0e8b099024c6a923cc</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>ArrayHandlerDDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>aec8696d377b0d9461dec6a440c1b8b15</anchor>
      <arglist>(std::function&lt; AL(const AR &amp;)&gt; copy_func)</arglist>
    </member>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a05a5d5f3a57548f6d5d6451dd4236172</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a40f21d8a1d3a4c9e3761f02d0d195a56</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a8d27b11bd2cbf656444310065daaaf90</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a9bdd2084e01645890ad067df4f557629</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a540ff543dbf816ade506e4cbac615325</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a519be01e94da0dbb6a69bf0a0f4fcc3e</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>afec5927acb5661806b8af05328e5be32</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a2a654c1a1e213a884b9d36b78a683102</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a27a49873388daafd66e5dad2a4212a21</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a4ec7f5bbe19dea40d3fd3bb27a06718e</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a4481d9881f5c20990b3db602e4da0698</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::function&lt; AL(const AR &amp;)&gt;</type>
      <name>m_copy_func</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDisk.html</anchorfile>
      <anchor>a311ece305fc0afcf359911dd69816cf9</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDDiskDistr</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</filename>
    <templarg>class AL</templarg>
    <templarg>class AR</templarg>
    <base>ArrayHandler&lt; AL, AL &gt;</base>
    <member kind="function">
      <type></type>
      <name>ArrayHandlerDDiskDistr</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>a023f4f9014ef95ec98713a1653c6df33</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>ArrayHandlerDDiskDistr</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>afd29c19c7ec8c30ac43b634fc6ae0b84</anchor>
      <arglist>(std::function&lt; AL(const AR &amp;)&gt; copy_func)</arglist>
    </member>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>ae30c74995215dabf2fcb8950ccfe7440</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>a3fc57bf0eaa3bf00a4622c647b25371e</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>a9894e0fc39586b9cb98a95d448d623b5</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>a79e8c39d4038b7b019acdd09509c0bf0</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>ac38788b8585593909b3c6767b1d18ddb</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>a4dbc42541ca5526f5bfd4da673c4d49c</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>af8861bba8dadd34376e9692ea6557c28</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>a95cf2353b90cf63b7092a6bdb9187ac6</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>a3f5c0918f4c61cae920e36c8b5f2f9a1</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>adcf36a2ecdd98fea019b790164cd0975</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>ab0df5b05835057027d07c7bb39a7b9db</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::function&lt; AL(const AR &amp;)&gt;</type>
      <name>m_copy_func</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskDistr.html</anchorfile>
      <anchor>a9b69e3bde33e878efae97529298f2be5</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDDiskSparse</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse.html</filename>
    <templarg>typename AL</templarg>
    <templarg>typename AR</templarg>
    <templarg>bool</templarg>
    <base>ArrayHandler&lt; AL, AR &gt;</base>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDDiskSparse&lt; AL, AR, true &gt;</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</filename>
    <templarg>typename AL</templarg>
    <templarg>typename AR</templarg>
    <base>ArrayHandler&lt; AL, AR &gt;</base>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>aad9ee4ff8b6ab110e4d46eba2f04c11e</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a694a450e398f2fb127c790b4c27cf7a3</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a4bea0407aecec45a5a70b089aec8b2fb</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>ab073221020adcc815bc4f9291b94420d</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>afb4910ee4aa57d862ee8408de7e3cdca</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>ac0fcd9e78299fe6195723d5b52c5e416</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a8d31fc0225de9acd06e2e59fc9c3b2b3</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>af62513e6d6ae0c8f490bbe3f6d28cbeb</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a372f5f9b69d3d0941b9919df7843814a</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a13fcb77b01221daea996518992145f53</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDDiskSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a15b9a885e2a2c546ca72dacbcee60f9d</anchor>
      <arglist>() override</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDefault</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</filename>
    <templarg>class AL</templarg>
    <templarg>class AR</templarg>
    <base>ArrayHandler&lt; AL, AL &gt;</base>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>a12fc26bd35a7ba1db11d86842ccbb1e9</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>a785a4804107d270bf97bfef0be11b86c</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>a95f8a6b4be76f8e5b3fdf39ac19fa317</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>a4edc618f5ca44d0374f359df171ec797</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>a8064e8c48760323e40b3b8929481d5a7</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>afd4a21532890d418db80aa3d1a974e1f</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>aecb9a99f4e55af2bb59074157ce50701</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>a081330057ef9b279b0e03404f7ae1b6b</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>a36f003d201c1ecc7a7e57d3d9b893c74</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>a0c6e3353980bcf0ec99d7b8dbce43219</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDefault.html</anchorfile>
      <anchor>aaa26e35138657a3dbd47855fef336884</anchor>
      <arglist>() override</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDistr</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</filename>
    <templarg>class AL</templarg>
    <templarg>class AR</templarg>
    <base>ArrayHandler&lt; AL, AL &gt;</base>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>a21f21442bc3dc5f06849a8f3f719cb8d</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>a5adb92bb4c737f4d870cd45972b7ea76</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>a56b3ef7d07556f9381e21212ea107076</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>ad32067cd0331494fc2692562cd9b15a4</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>aceea181a8bcd435b070a35b2dfbd67ee</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>ab68a3f85e09928e1e49268cc37cef9d2</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>af00b3c3617a07fcaa14f69f301237fba</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>aee239aa422ee5263315a0d54a1fdec8a</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>a9d49ce8dfd443418e73e669f7b022e54</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>a4880520968fb9b52ba8a040cc930b06e</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistr.html</anchorfile>
      <anchor>a9f8d6eb8d0538129a3e28700bed3a17f</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDistrDDisk</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</filename>
    <templarg>class AL</templarg>
    <templarg>class AR</templarg>
    <base>ArrayHandler&lt; AL, AL &gt;</base>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>ae6db0ab25995d1e01fb5c89065ac4dcb</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>a4a073af506cab23505a4b12eeb304092</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>aae260a18605448774355afc669d47dba</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>aaa48f1ed5d7d568b1bf0e761528b0b60</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>a3ad1dab6c3d9845044cf0d646ec9b8d9</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>a669acdf49cbf4ce55ba0b56e28de2293</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>a020a538eea512191129c6d7532263fce</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>a2acefebaf2757533a74bdce0749fde3a</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>a339518c7f1b8863edc650c41bf391985</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>a4f8aaaa1a6b534d84727a0244737cc56</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrDDisk.html</anchorfile>
      <anchor>ad5d320abc83b0bcfa900f6e1fc28925f</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDistrSparse</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse.html</filename>
    <templarg>typename AL</templarg>
    <templarg>typename AR</templarg>
    <templarg>bool</templarg>
    <base>ArrayHandler&lt; AL, AR &gt;</base>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerDistrSparse&lt; AL, AR, true &gt;</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</filename>
    <templarg>typename AL</templarg>
    <templarg>typename AR</templarg>
    <base>ArrayHandler&lt; AL, AR &gt;</base>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>ae1f29f99a303ac2efe4d24bc550d92fa</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a2525b401ae39fdf5a4488d5865b392c2</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a1964cee0a09332647048a161de77c7fa</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a0930b94c425de8db03ae264089cfe1aa</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>aabd1dc42a41f54d0a3bf81ce90004671</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a716230e8181659417de7abaf8f2be1c9</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a81cfa268825271be8554d65a401f5f49</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>aae8ca880817562a32b11937fe2b6528b</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a9a9b379d0cd817fd2947f815ceb96ed2</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>adc8dae9f963bcd0af78e57a68943dbff</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerDistrSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a08579ecc6d34583cb53f7d84a4767ba8</anchor>
      <arglist>() override</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::ArrayHandlerError</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1ArrayHandlerError.html</filename>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerIterable</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</filename>
    <templarg>typename AL</templarg>
    <templarg>typename AR</templarg>
    <base>ArrayHandler&lt; AL, AL &gt;</base>
    <member kind="function">
      <type></type>
      <name>ArrayHandlerIterable</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>a0543b35992ac0ab181a14579dd55466d</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>ArrayHandlerIterable</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>ab4436c307acc2a8497d485cc97d7daed</anchor>
      <arglist>(const ArrayHandlerIterable&lt; AL, AR &gt; &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>a227ca461cbb92a7a17139f9c197c8b20</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>a8e347dcac62a193a35b1c52d437ea8de</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>a5db780196a19353b348d8796e4deef1f</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>ab9319fe0821865225f9228c260aea18d</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>af90e9c4e402d165a690d98d9539b45af</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>ac3ffaf7e2d152e965cf34bee3e470f21</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>a7a7b2a8d19bd6af3972ed9d3c98e7714</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>aee3831a407bb5c0682f72243c96826d7</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>aaffe2da9e7a6add977601befc3ed3154</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>ae1c852bf2a6b01fe1eee6a267fc5c293</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>a1d0a3ef3438f7b60b7e920a905acc396</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>T</type>
      <name>copyAny</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>a16b328328cac8d8cbfccf744fa7baa4f</anchor>
      <arglist>(const S &amp;source)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>T</type>
      <name>copyAny</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterable.html</anchorfile>
      <anchor>a16b328328cac8d8cbfccf744fa7baa4f</anchor>
      <arglist>(const S &amp;source)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerIterableSparse</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse.html</filename>
    <templarg>typename AL</templarg>
    <templarg>typename AR</templarg>
    <templarg>bool</templarg>
    <base>ArrayHandler&lt; AL, AR &gt;</base>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerIterableSparse&lt; AL, AR, true &gt;</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</filename>
    <templarg>typename AL</templarg>
    <templarg>typename AR</templarg>
    <base>ArrayHandler&lt; AL, AR &gt;</base>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a553427e899b8944992fc7f4cae9134a9</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a6a279a802ab150a11edf34b62e823dd0</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a04d7ff27152cf612890cda08ad68786d</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a6f91bc41da1e56a704c3caeeba99c8b5</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a6227a2692295431e70e36770b825d1c9</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>aac419790200495724fad84d6d61bb471</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a0512e93ed50ecb56411f79740cc9dc68</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>addf241c8a7c0a7e35dd92bbb9c5040ca</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a518106fc3e92887fd0d538405c46ead2</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>a0f96fe9439116a42c8e9a3e652f6a649</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerIterableSparse_3_01AL_00_01AR_00_01true_01_4.html</anchorfile>
      <anchor>ad5c7e5e879b0515d803956aa562e0324</anchor>
      <arglist>() override</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::ArrayHandlers</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</filename>
    <templarg>typename R</templarg>
    <templarg>typename Q</templarg>
    <templarg>typename P</templarg>
    <class kind="class">molpro::linalg::itsolv::ArrayHandlers::Builder</class>
    <member kind="function">
      <type></type>
      <name>ArrayHandlers</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>aaf5be4e330e1306c4df3d0a12d91786e</anchor>
      <arglist>(std::shared_ptr&lt; array::ArrayHandler&lt; R, R &gt; &gt; rr, std::shared_ptr&lt; array::ArrayHandler&lt; Q, Q &gt; &gt; qq, std::shared_ptr&lt; array::ArrayHandler&lt; P, P &gt; &gt; pp, std::shared_ptr&lt; array::ArrayHandler&lt; R, Q &gt; &gt; rq, std::shared_ptr&lt; array::ArrayHandler&lt; R, P &gt; &gt; rp, std::shared_ptr&lt; array::ArrayHandler&lt; Q, R &gt; &gt; qr, std::shared_ptr&lt; array::ArrayHandler&lt; Q, P &gt; &gt; qp)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>ArrayHandlers</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>adcc4ea6c5d53db828f4953f57e583465</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto &amp;</type>
      <name>rr</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>ada83676611311f4d5a22dedaed158d19</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto &amp;</type>
      <name>qq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a549e3557ff2d1348ed9de0fc91841612</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto &amp;</type>
      <name>pp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a911075252882d6819b4f3e98ffc618df</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto &amp;</type>
      <name>rq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>aef0571a856c054f32b6a0eb417e8a28a</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto &amp;</type>
      <name>qr</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a06cc917ff69b072ee3025815654ed73f</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto &amp;</type>
      <name>rp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>ad1e5871229fc21d462b6d5e42a045e70</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto &amp;</type>
      <name>qp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>aa264c3c28c97bc6b83e0301f465804b6</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static Builder</type>
      <name>create</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a511c639fd98850e5e6c294595bbeff50</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; R, R &gt; &gt;</type>
      <name>m_rr</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a7ae3618cda4c0dc9e566f9fbfb03c026</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; Q, Q &gt; &gt;</type>
      <name>m_qq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>ab9334bdef83e3aa16d9b3df513a333e4</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; P, P &gt; &gt;</type>
      <name>m_pp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a25bd49b99803043b017fd2480b67ab3e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; R, Q &gt; &gt;</type>
      <name>m_rq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a7afb128736a83613187fe950bad8672a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; R, P &gt; &gt;</type>
      <name>m_rp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>af4c69a0d6a42b735725ff861d166dcc2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; Q, R &gt; &gt;</type>
      <name>m_qr</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a0e9fbedc8c1eaed9408d27e1e9029bf2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; Q, P &gt; &gt;</type>
      <name>m_qp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers.html</anchorfile>
      <anchor>a07cf5b2078fd7c5c58158682d4a61d52</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::util::ArrayHandlersError</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1util_1_1ArrayHandlersError.html</filename>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandlerSparse</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</filename>
    <templarg>typename AL</templarg>
    <templarg>typename AR</templarg>
    <base>ArrayHandler&lt; AL, AL &gt;</base>
    <member kind="function">
      <type>AL</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>ae9bd96ab8b7da7a151110e536ef0eba8</anchor>
      <arglist>(const AR &amp;source) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>a946bb288aa823fe13ac75acac405dffa</anchor>
      <arglist>(AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>a8368510447f4dc4794b20b3798874508</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>a878972b64b1a7e66d67cc3d75a9636bb</anchor>
      <arglist>(value_type alpha, AL &amp;x) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>aa6221fc7c2a7fe3675383fa6bbceba8f</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>a108c43c4d2933e6b3a2b8d3097ddb2a3</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>aa1b452951e50cf58e8056e158b27df57</anchor>
      <arglist>(const Matrix&lt; value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>gemm_inner</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>af2b1583877b672acaf8797edc9650c21</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type_abs &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>aab9f4777b83c10dc050d77e84cbd9c66</anchor>
      <arglist>(size_t n, const AL &amp;x, const AR &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>a4616ed40513897847357978028405171</anchor>
      <arglist>(size_t n, const AL &amp;x, bool max=false, bool ignore_sign=false) override</arglist>
    </member>
    <member kind="function">
      <type>ProxyHandle</type>
      <name>lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandlerSparse.html</anchorfile>
      <anchor>ad8d571c5f450ec725daf8ae633e746e6</anchor>
      <arglist>() override</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::util::BufferManager</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</filename>
    <templarg>class T</templarg>
    <class kind="struct">molpro::linalg::array::util::BufferManager::Iterator</class>
    <member kind="function">
      <type></type>
      <name>BufferManager</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>abfadf83671b356bd738080ed952e23b5</anchor>
      <arglist>(const CVecRef &amp;arrays, size_t buffer_size=8192, int number_of_buffers=2)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>BufferManager</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a92d6ea724ca363570fd0e6c3fe999e8f</anchor>
      <arglist>(const std::vector&lt; T &gt; &amp;arrays, size_t buffer_size=8192, int number_of_buffers=2)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>BufferManager</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>ac5bf2d67fde9f5d4697dda772b8eb7d8</anchor>
      <arglist>(const T &amp;array, size_t buffer_size=8192, int number_of_buffers=2)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>buffer_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a6e8bcda1929569fd5328415d31b5eec6</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>buffer_stride</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a7eb7ac21d7f7c2b1aba452a5bf4869b2</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>buffer_offset</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a17ce61870f54ec4b586d26f07cadccc2</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Iterator</type>
      <name>begin</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>ac79be08b05f977d56fc944fb7302836e</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>Iterator</type>
      <name>end</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a9095a1d96c2d44d2202aca4c3f7265be</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="typedef" protection="protected">
      <type>typename T::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a3afacd38e3d5c415e325e8b25f560a34</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" protection="protected">
      <type>Span&lt; value_type &gt;</type>
      <name>next</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a9d567f6ee13c27e6e11fd9ba51ac2eef</anchor>
      <arglist>(bool initial=false)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>const CVecRef &amp;</type>
      <name>m_arrays</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a2b25503f73cc1c73697ef0b91d520861</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>const size_t</type>
      <name>m_buffer_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>ab10d8d1e9817dec589cd6e9184363669</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>const int</type>
      <name>m_number_of_buffers</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>adc68c0146902898f004096134c6c6b74</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>size_t</type>
      <name>m_current_buffer_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>aabab9550ecab3a412f9f17199c48a288</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>size_t</type>
      <name>m_current_segment</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a57d7917ce1a5d3c51c9ba42e6b2c550c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>const std::pair&lt; size_t, size_t &gt;</type>
      <name>m_range</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>a65dcf83009499d33b40a0904d61b1f84</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; typename T::value_type &gt;</type>
      <name>m_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>ae51796188e203f326af9e58462506cd0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::future&lt; void &gt;</type>
      <name>m_read_future</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager.html</anchorfile>
      <anchor>ac8769c67a903636dfe9202c0c668ffa7</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::ArrayHandlers::Builder</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</filename>
    <class kind="class">molpro::linalg::itsolv::ArrayHandlers::Builder::Proxy</class>
    <member kind="function">
      <type></type>
      <name>Builder</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>a7ea91083ab19617ac240aa7720a5b14d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>ArrayHandlers</type>
      <name>build</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>abb67948c3c5053c20c411b94dfa84a5d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable">
      <type>Proxy&lt; R, R &gt;</type>
      <name>rr</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>a9ccab0aa513fb9d0b4653015fe5e4a64</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Proxy&lt; Q, Q &gt;</type>
      <name>qq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>a2be277beaf97a327fa797d46253cca3b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Proxy&lt; P, P &gt;</type>
      <name>pp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>a297b6cbd16b811019b4e3bb3a5a9702a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Proxy&lt; R, Q &gt;</type>
      <name>rq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>aa6981400397ab1a29363b413580827e0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Proxy&lt; R, P &gt;</type>
      <name>rp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>a378060b02cf92fcf204b2398ff336f5d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Proxy&lt; Q, R &gt;</type>
      <name>qr</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>ad939522fa33168f18c57e576cd134e75</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Proxy&lt; Q, P &gt;</type>
      <name>qp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder.html</anchorfile>
      <anchor>ae2bfc2e0ce3f1abeea46cfb576577d50</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::CastOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</filename>
    <member kind="function" static="yes">
      <type>static std::shared_ptr&lt; LinearEigensystemDavidsonOptions &gt;</type>
      <name>LinearEigensystem</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a57216e0e890458b1477db3e35d66190e</anchor>
      <arglist>(const std::shared_ptr&lt; Options &gt; &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static const LinearEigensystemDavidsonOptions &amp;</type>
      <name>LinearEigensystem</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a9537fd37eaf8c1e28e296552aaaf1ab6</anchor>
      <arglist>(const Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static const LinearEigensystemRSPTOptions &amp;</type>
      <name>LinearEigensystemRSPT</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a89c0ee00288f2a49719b4218dd89ad88</anchor>
      <arglist>(const Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static LinearEigensystemDavidsonOptions &amp;</type>
      <name>LinearEigensystem</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a35d6a23a0fc8a74555d3879661c7ea8a</anchor>
      <arglist>(Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static std::shared_ptr&lt; LinearEquationsDavidsonOptions &gt;</type>
      <name>LinearEquations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>afdfd19f07b66c2db44292178439a3e89</anchor>
      <arglist>(const std::shared_ptr&lt; Options &gt; &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static const LinearEquationsDavidsonOptions &amp;</type>
      <name>LinearEquations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>af5bff51ad342e98ed1178fb7c533ec19</anchor>
      <arglist>(const Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static LinearEquationsDavidsonOptions &amp;</type>
      <name>LinearEquations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>ace5b27f7d2a1a5523d9cc8bb9608ea11</anchor>
      <arglist>(Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static std::shared_ptr&lt; NonLinearEquationsDIISOptions &gt;</type>
      <name>NonLinearEquationsDIIS</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a88d9b947bdc350900491faf6667a6d06</anchor>
      <arglist>(const std::shared_ptr&lt; Options &gt; &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static const NonLinearEquationsDIISOptions &amp;</type>
      <name>NonLinearEquationsDIIS</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a28e1092b3a62d74633ddcb1c4f3c6f16</anchor>
      <arglist>(const Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static NonLinearEquationsDIISOptions &amp;</type>
      <name>NonLinearEquationsDIIS</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a992b987cf7c686568b74025c99edbd19</anchor>
      <arglist>(Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static std::shared_ptr&lt; OptimizeBFGSOptions &gt;</type>
      <name>OptimizeBFGS</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>aa89b51d3176f70cb6f6ef3c6068ed9b2</anchor>
      <arglist>(const std::shared_ptr&lt; Options &gt; &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static const OptimizeBFGSOptions &amp;</type>
      <name>OptimizeBFGS</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a161dcf3fd2961f3f89cdfad943bdeafd</anchor>
      <arglist>(const Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static OptimizeBFGSOptions &amp;</type>
      <name>OptimizeBFGS</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a7ba71635161687c4af6a8970c15ef423</anchor>
      <arglist>(Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static std::shared_ptr&lt; OptimizeSDOptions &gt;</type>
      <name>OptimizeSD</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>af464a34dd6d6c67df4673fe37dba677b</anchor>
      <arglist>(const std::shared_ptr&lt; Options &gt; &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static const OptimizeSDOptions &amp;</type>
      <name>OptimizeSD</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>ae9eeb50c178e2c6a09c07edb75f1364d</anchor>
      <arglist>(const Options &amp;options)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static OptimizeSDOptions &amp;</type>
      <name>OptimizeSD</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1CastOptions.html</anchorfile>
      <anchor>a2983da708a14b8539df50b5c81e0b099</anchor>
      <arglist>(Options &amp;options)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::CompareAbs</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1CompareAbs.html</filename>
    <templarg>typename T</templarg>
    <templarg>class Compare</templarg>
    <member kind="function">
      <type>constexpr bool</type>
      <name>operator()</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1CompareAbs.html</anchorfile>
      <anchor>a25665cf151141173b8b4ec3562a38c99</anchor>
      <arglist>(const T &amp;lhs, const T &amp;rhs) const</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::ArrayHandler::Counter</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1Counter.html</filename>
    <member kind="variable">
      <type>int</type>
      <name>scal</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1Counter.html</anchorfile>
      <anchor>a9d520e242d7fcaaa775768eaf9a834dc</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>dot</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1Counter.html</anchorfile>
      <anchor>a5ad2b6f45fd33ca55ec7c1b110f2066e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>axpy</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1Counter.html</anchorfile>
      <anchor>a79929e52bd35c9046eb3e9dca6f2b71f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>copy</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1Counter.html</anchorfile>
      <anchor>a6241ac3f1e9f88d16def846e3d5ed5f4</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>gemm_inner</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1Counter.html</anchorfile>
      <anchor>a5cbd61f446c2aaf730f319376be95331</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>gemm_outer</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1Counter.html</anchorfile>
      <anchor>a81a28a4bc061af75fec23768979e1bc2</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::detail::create_default_handler</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1detail_1_1create__default__handler.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <templarg>typename</templarg>
    <member kind="function">
      <type>auto</type>
      <name>operator()</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1detail_1_1create__default__handler.html</anchorfile>
      <anchor>a0af627edca1a5b77733b8fa65edb0da4</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::Matrix::CSlice</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</filename>
    <member kind="function">
      <type></type>
      <name>CSlice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</anchorfile>
      <anchor>a68078ec035cbdfed08d2196724879980</anchor>
      <arglist>(const Matrix&lt; T &gt; &amp;matrix, coord_type upper_left, coord_type bottom_right)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>CSlice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</anchorfile>
      <anchor>aabcd5f985dca4d0defdb7df9da0dfb10</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~CSlice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</anchorfile>
      <anchor>a42b52869b5d28ce2053dc36cd7743d94</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>CSlice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</anchorfile>
      <anchor>ae707f005edd4908b43b90c9d8f71e9bb</anchor>
      <arglist>(const CSlice &amp;)=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>CSlice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</anchorfile>
      <anchor>a07a163a9a70be7cc8b401ab1e38ae500</anchor>
      <arglist>(CSlice &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>CSlice &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</anchorfile>
      <anchor>ab86e8200c38a955e1f1cef06ade90b92</anchor>
      <arglist>(CSlice &amp;&amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>T</type>
      <name>operator()</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</anchorfile>
      <anchor>a458f08296bae9a3126991c7084c12dbc</anchor>
      <arglist>(size_t i, size_t j) const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Slice</type>
      <name>m_slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1CSlice.html</anchorfile>
      <anchor>a266d4c925653909a69fb71ced5b3cdcf</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::decay</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1decay.html</filename>
    <templarg>class T</templarg>
    <member kind="typedef">
      <type>std::decay_t&lt; T &gt;</type>
      <name>type</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1decay.html</anchorfile>
      <anchor>a1b8684ae6299300c17f9bf76d5006df2</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::decay&lt; std::reference_wrapper&lt; T &gt; &gt;</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1decay_3_01std_1_1reference__wrapper_3_01T_01_4_01_4.html</filename>
    <templarg>class T</templarg>
    <member kind="typedef">
      <type>std::decay_t&lt; T &gt;</type>
      <name>type</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1decay_3_01std_1_1reference__wrapper_3_01T_01_4_01_4.html</anchorfile>
      <anchor>a789ae045715e1ffd6d05755295fba091</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <templarg>ArrayFamily</templarg>
    <templarg>ArrayFamily</templarg>
    <member kind="typedef">
      <type>ArrayHandlerDefault&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler.html</anchorfile>
      <anchor>aec8251d9a4a1d7a2bb3e9a2ea601775c</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Distributed, ArrayFamily::Distributed &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributeddbf8f5ddddbad731ea3a8cbaa1b36776.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerDistr&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributeddbf8f5ddddbad731ea3a8cbaa1b36776.html</anchorfile>
      <anchor>a690900049363b1b4c77c00154c296968</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Distributed, ArrayFamily::DistributedDisk &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributedbd43e8f0864faf7cff59f1977d87f421.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerDistrDDisk&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributedbd43e8f0864faf7cff59f1977d87f421.html</anchorfile>
      <anchor>a2197ae85b68a26480bf2aa88aa594638</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Distributed, ArrayFamily::Sparse &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributed48d18a9ac3c5b993ea3611b2e8025a0c.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerDistrSparse&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributed48d18a9ac3c5b993ea3611b2e8025a0c.html</anchorfile>
      <anchor>aafb7437efd9196d87749a391dcc46753</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::DistributedDisk, ArrayFamily::Distributed &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributed3dcbef7436a9781ba51119693d448467.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerDDiskDistr&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributed3dcbef7436a9781ba51119693d448467.html</anchorfile>
      <anchor>a7cadadbafbc1b85bf9fd27080887a07a</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::DistributedDisk, ArrayFamily::DistributedDisk &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributed68a38ca64bb82db94f7881986d11feac.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerDDisk&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributed68a38ca64bb82db94f7881986d11feac.html</anchorfile>
      <anchor>ae41c24b833191cd49c0552edc689c4e1</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::DistributedDisk, ArrayFamily::Sparse &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributed36aa416066f57f33fdac6a4fbd137692.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerDDiskSparse&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Distributed36aa416066f57f33fdac6a4fbd137692.html</anchorfile>
      <anchor>add88cf6c11c49b82df633f3672ce078b</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Iterable, ArrayFamily::Iterable &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Iterable_00_01ArrayFamily_1_1Iterable_01_4.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerIterable&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Iterable_00_01ArrayFamily_1_1Iterable_01_4.html</anchorfile>
      <anchor>a51e27e7ebd4d8f4da742bf04306e2548</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Iterable, ArrayFamily::Sparse &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Iterable_00_01ArrayFamily_1_1Sparse_01_4.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerIterableSparse&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Iterable_00_01ArrayFamily_1_1Sparse_01_4.html</anchorfile>
      <anchor>a0b6b87438b26d718033fd48547c3e991</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Sparse, ArrayFamily::Sparse &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Sparse_00_01ArrayFamily_1_1Sparse_01_4.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="typedef">
      <type>ArrayHandlerSparse&lt; T, S &gt;</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1default__handler_3_01T_00_01S_00_01ArrayFamily_1_1Sparse_00_01ArrayFamily_1_1Sparse_01_4.html</anchorfile>
      <anchor>aa117fcd11491ae454c7dc220edcaed16</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::Dimensions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</filename>
    <member kind="function">
      <type></type>
      <name>Dimensions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>aed2dc5307107fb8833b44e587a101010</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Dimensions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>a0c38435c2de283433116bf47f814361c</anchor>
      <arglist>(size_t np, size_t nq, size_t nc)</arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>nP</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>a6f870dc7e7e5f83f02137b6dab4a2060</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>nQ</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>a46dc5debc430f86a816804fd92f74083</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>nD</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>a154456f7740b0eb6fa70f2ff8a02d74f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>nX</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>a28e3ec823f95c30ae8d28bab53076a3a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>oP</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>a8b2d29e64f9ba7bb0ceddbba9e0e4632</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>oQ</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>a06259f6b0e7d2a7298b78186c869d593</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>oD</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>aa019fa3b705f49ce4e180221bf57398c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>nRHS</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Dimensions.html</anchorfile>
      <anchor>a4befc78adaf6a673c6bb6592ac661061</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::DistrArray</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</filename>
    <class kind="class">molpro::linalg::array::DistrArray::LocalBuffer</class>
    <member kind="typedef">
      <type>void</type>
      <name>distributed_array</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a134005a9cacb96ab592211c9c5e75d77</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>double</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a01ce15f1218bd98d16a76447b98d40cd</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>size_t</type>
      <name>index_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ada7d77229bb5b1ed17256c5407ff5702</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::map&lt; index_type, double &gt;</type>
      <name>SparseArray</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a3fcf79ae1da7f58c1ddbe71f90042f75</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>util::Distribution&lt; index_type &gt;</type>
      <name>Distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a1ff2572f6537752365b0d0354015a850</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~DistrArray</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a9429b74abc625dd5097ae93c4ed1b87a</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type>MPI_Comm</type>
      <name>communicator</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ab53011e0965e3ddee49a8588a2d623de</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>sync</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>aa52a35fbcda9f9d6bb66dad71d6c1ab8</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ab2146249e167e65d8b5204bf2e25a154</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>compatible</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a1b46632636f9f92c9f25bebcc4c71230</anchor>
      <arglist>(const DistrArray &amp;other) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>zero</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a08bbd67ad5c2d9dc2319f45abf3663fd</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>error</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>abea8eec2c55f186b7ae9fbc3ffd2f8f5</anchor>
      <arglist>(const std::string &amp;message) const</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>operator[]</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>abe2b8d71f06ec792bef8887f28cc6c80</anchor>
      <arglist>(size_t index)</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::unique_ptr&lt; LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a6eddd4b6d9683979c764db15d4aba048</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::unique_ptr&lt; const LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>acc95a3e905bd982b22ae527e00393c0c</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const Distribution &amp;</type>
      <name>distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a673acdaae53ad3ed20d56bd4f493eaa3</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual value_type</type>
      <name>at</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ad22625b1ebc1d8b3b9b608503f722228</anchor>
      <arglist>(index_type ind) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a49248193a008caa469284ecd577bbfb6</anchor>
      <arglist>(index_type ind, value_type val)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a315204c97a4b1eac89f03bfb5b654f9a</anchor>
      <arglist>(index_type lo, index_type hi, value_type *buf) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::vector&lt; value_type &gt;</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a3d6f5fc27937fe8283b837ea9b2b80c7</anchor>
      <arglist>(index_type lo, index_type hi) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>put</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a325eae41a73eda5f5d42570dd7d75e49</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ada2a0142ff296e878e17cc78655bd6e7</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::vector&lt; value_type &gt;</type>
      <name>gather</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>aa271bbd89f844c208f8cc32b3a33d2bc</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>scatter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>adee19c5a4930dec6e5564d7b37bf9d3a</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>scatter_acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a7d5c587fb50d97d9e20559707f0d5283</anchor>
      <arglist>(std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::vector&lt; value_type &gt;</type>
      <name>vec</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a2b3c8e9c3691b02452086960db32f007</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a0e5fc760719f8af338fc47c2d5cc5352</anchor>
      <arglist>(value_type val)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a13953b59c18f19d9b2afaf024f0363d9</anchor>
      <arglist>(const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>copy_patch</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ad416250c4a3eb8b72999ccdc8fe2b253</anchor>
      <arglist>(const DistrArray &amp;y, index_type start, index_type end)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a2599f5d68e2be0d8dcc5c5cc01778d2d</anchor>
      <arglist>(value_type a, const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a63ab08569b888803190980dccf6db5cb</anchor>
      <arglist>(value_type a, const SparseArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>aa5b727d10d41e677fefcfe4c92de2d5e</anchor>
      <arglist>(value_type a)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>add</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>aca2e59deac50295a278e8ea11253d994</anchor>
      <arglist>(const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>add</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a9ca27d6587a5fb6acf6314f2f842d026</anchor>
      <arglist>(value_type a)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>sub</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ae319b68c20b5d17756a54c0577535117</anchor>
      <arglist>(const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>sub</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a3b49bf771e7b02f5e1200376bf52ae0e</anchor>
      <arglist>(value_type a)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>recip</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a122eacfba5a5234932878dbfb1ddf61d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>times</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a71816fdad079e523e082b2e650fc1249</anchor>
      <arglist>(const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>times</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a1d47c67f719dd4da6260ce545336da7f</anchor>
      <arglist>(const DistrArray &amp;y, const DistrArray &amp;z)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a4d32f113660ca10a15f2bea070389799</anchor>
      <arglist>(const DistrArray &amp;y) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ac25fdc7e140b903be31399f2c352caa7</anchor>
      <arglist>(const SparseArray &amp;y) const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>divide</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ab76e40bda340ceb59a2c816054d64223</anchor>
      <arglist>(const DistrArray &amp;y, const DistrArray &amp;z, value_type shift=0, bool append=false, bool negative=false)</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; index_type, value_type &gt; &gt;</type>
      <name>min_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a2545957d3297b23695193a23b5f5c583</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; index_type, value_type &gt; &gt;</type>
      <name>max_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a3b82e67df9be2a3d3a54b7ec918f8774</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; index_type, value_type &gt; &gt;</type>
      <name>min_abs_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a53135a777219713f48d04eff91a21ac1</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; index_type, value_type &gt; &gt;</type>
      <name>max_abs_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>af4849eb784d0523371f0f4734a2bdba7</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; index_type &gt;</type>
      <name>min_loc_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>acb46e64c58bed7fc1336ceab608c4d4d</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a182c67ba7775724fa72bbf57d924aa43</anchor>
      <arglist>(size_t n, const DistrArray &amp;y) const</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a46de5bc51eb99248d4610d0c7724d15f</anchor>
      <arglist>(size_t n, const SparseArray &amp;y) const</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a5cf3478a5c32364c0701ce1f2a0c43b6</anchor>
      <arglist>(size_t n, bool max=false, bool ignore_sign=false) const</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>DistrArray</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a06cddc4939b74533794457dd961d91f9</anchor>
      <arglist>(size_t dimension, MPI_Comm commun)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>DistrArray</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>aca5fd27c1376c9edf2079a156a58fb87</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual void</type>
      <name>_divide</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a04c5f6eabcf33711aaa8df2b0297a5e1</anchor>
      <arglist>(const DistrArray &amp;y, const DistrArray &amp;z, value_type shift, bool append, bool negative)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>index_type</type>
      <name>m_dimension</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a7bb623217ec8c544e3f47b4657469638</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>MPI_Comm</type>
      <name>m_communicator</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a2a4ce8069d27031d26fad2ad7e4671dd</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::unique_ptr&lt; LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a6eddd4b6d9683979c764db15d4aba048</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::unique_ptr&lt; const LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>acc95a3e905bd982b22ae527e00393c0c</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const Distribution &amp;</type>
      <name>distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a673acdaae53ad3ed20d56bd4f493eaa3</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual value_type</type>
      <name>at</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ad22625b1ebc1d8b3b9b608503f722228</anchor>
      <arglist>(index_type ind) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a49248193a008caa469284ecd577bbfb6</anchor>
      <arglist>(index_type ind, value_type val)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a315204c97a4b1eac89f03bfb5b654f9a</anchor>
      <arglist>(index_type lo, index_type hi, value_type *buf) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::vector&lt; value_type &gt;</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a3d6f5fc27937fe8283b837ea9b2b80c7</anchor>
      <arglist>(index_type lo, index_type hi) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>put</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a325eae41a73eda5f5d42570dd7d75e49</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ada2a0142ff296e878e17cc78655bd6e7</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::vector&lt; value_type &gt;</type>
      <name>gather</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>aa271bbd89f844c208f8cc32b3a33d2bc</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>scatter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>adee19c5a4930dec6e5564d7b37bf9d3a</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>scatter_acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a7d5c587fb50d97d9e20559707f0d5283</anchor>
      <arglist>(std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::vector&lt; value_type &gt;</type>
      <name>vec</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a2b3c8e9c3691b02452086960db32f007</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a0e5fc760719f8af338fc47c2d5cc5352</anchor>
      <arglist>(value_type val)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a13953b59c18f19d9b2afaf024f0363d9</anchor>
      <arglist>(const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>copy_patch</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ad416250c4a3eb8b72999ccdc8fe2b253</anchor>
      <arglist>(const DistrArray &amp;y, index_type start, index_type end)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a2599f5d68e2be0d8dcc5c5cc01778d2d</anchor>
      <arglist>(value_type a, const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a63ab08569b888803190980dccf6db5cb</anchor>
      <arglist>(value_type a, const SparseArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>aa5b727d10d41e677fefcfe4c92de2d5e</anchor>
      <arglist>(value_type a)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>add</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>aca2e59deac50295a278e8ea11253d994</anchor>
      <arglist>(const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>add</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a9ca27d6587a5fb6acf6314f2f842d026</anchor>
      <arglist>(value_type a)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>sub</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ae319b68c20b5d17756a54c0577535117</anchor>
      <arglist>(const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>sub</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a3b49bf771e7b02f5e1200376bf52ae0e</anchor>
      <arglist>(value_type a)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>recip</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a122eacfba5a5234932878dbfb1ddf61d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>times</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a71816fdad079e523e082b2e650fc1249</anchor>
      <arglist>(const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>times</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a1d47c67f719dd4da6260ce545336da7f</anchor>
      <arglist>(const DistrArray &amp;y, const DistrArray &amp;z)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a4d32f113660ca10a15f2bea070389799</anchor>
      <arglist>(const DistrArray &amp;y) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ac25fdc7e140b903be31399f2c352caa7</anchor>
      <arglist>(const SparseArray &amp;y) const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>divide</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>ab76e40bda340ceb59a2c816054d64223</anchor>
      <arglist>(const DistrArray &amp;y, const DistrArray &amp;z, value_type shift=0, bool append=false, bool negative=false)</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; index_type, value_type &gt; &gt;</type>
      <name>min_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a2545957d3297b23695193a23b5f5c583</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; index_type, value_type &gt; &gt;</type>
      <name>max_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a3b82e67df9be2a3d3a54b7ec918f8774</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; index_type, value_type &gt; &gt;</type>
      <name>min_abs_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a53135a777219713f48d04eff91a21ac1</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; index_type, value_type &gt; &gt;</type>
      <name>max_abs_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>af4849eb784d0523371f0f4734a2bdba7</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; index_type &gt;</type>
      <name>min_loc_n</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>acb46e64c58bed7fc1336ceab608c4d4d</anchor>
      <arglist>(int n) const</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a182c67ba7775724fa72bbf57d924aa43</anchor>
      <arglist>(size_t n, const DistrArray &amp;y) const</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select_max_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a46de5bc51eb99248d4610d0c7724d15f</anchor>
      <arglist>(size_t n, const SparseArray &amp;y) const</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, value_type &gt;</type>
      <name>select</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray.html</anchorfile>
      <anchor>a5cf3478a5c32364c0701ce1f2a0c43b6</anchor>
      <arglist>(size_t n, bool max=false, bool ignore_sign=false) const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::DistrArrayDisk</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</filename>
    <base>molpro::linalg::array::DistrArray</base>
    <class kind="class">molpro::linalg::array::DistrArrayDisk::LocalBufferDisk</class>
    <member kind="typedef">
      <type>void</type>
      <name>disk_array</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a3b7d0a28a47e1f354506704c5050ec39</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>erase</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>aab106126cbeb07d587e448572a6bc2f0</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function">
      <type>const Distribution &amp;</type>
      <name>distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>ad485855dd9584378e097b2048f55c188</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>abc0c53e6bab3b4f900a3db33b8989fb2</anchor>
      <arglist>(const DistrArrayDisk &amp;y) const</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a7538f5e319160f6431ed995732f2f4bd</anchor>
      <arglist>(const DistrArray &amp;y) const override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>af247b4c8e347e5219b3ec9a142fd0145</anchor>
      <arglist>(const SparseArray &amp;y) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_buffer_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>af9f9964f531accee84aee648a0b55799</anchor>
      <arglist>(size_t buffer_size)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a12eff185b5102d1636287b09f269dadb</anchor>
      <arglist>(const DistrArray &amp;y) override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a5155710fe9ecf2caeb57961fee7eeb5a</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; const LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>ae3db2de9f782f8b17d47616aa1f30809</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>ae3d06b053dd86e22cce9bd220ec82bae</anchor>
      <arglist>(const span::Span&lt; value_type &gt; &amp;buffer)</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; const LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a485a9938423d0cdaec7b5b1aefac0743</anchor>
      <arglist>(const span::Span&lt; value_type &gt; &amp;buffer) const</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>DistrArrayDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a17a86cb94129f1e148d4848f467d9b6d</anchor>
      <arglist>(std::unique_ptr&lt; Distribution &gt; distr, MPI_Comm commun)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>DistrArrayDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a0ce0b6062c43af2ce8fe01528a6cc423</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>DistrArrayDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a817d74266ed831405aed81c778a83718</anchor>
      <arglist>(const DistrArray &amp;source)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>DistrArrayDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a48152a5319d58519cd9fcbfb4da61b5f</anchor>
      <arglist>(DistrArrayDisk &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>~DistrArrayDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a369fdeb6d5caa34edfd36c7ef9160a6f</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>DistrArray</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>a06cddc4939b74533794457dd961d91f9</anchor>
      <arglist>(size_t dimension, MPI_Comm commun)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>DistrArray</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>aca5fd27c1376c9edf2079a156a58fb87</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_allocated</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>aa2d8025003bcda8d27b6fe13f0a27dfe</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::unique_ptr&lt; Distribution &gt;</type>
      <name>m_distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>ae5b9c588896f146e816e2e67327fbeef</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>size_t</type>
      <name>m_buffer_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk.html</anchorfile>
      <anchor>ab7fafda24eade50953f30efa800e30b2</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::DistrArrayFile</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</filename>
    <base>molpro::linalg::array::DistrArrayDisk</base>
    <member kind="function">
      <type></type>
      <name>DistrArrayFile</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>ad81826a3275495c1445841a5fc028a0f</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayFile</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a3c8f8a75da0597a3b5e62690191f82d2</anchor>
      <arglist>(const DistrArrayFile &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayFile</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a944b7c4be9545e968dff05a23eb72134</anchor>
      <arglist>(DistrArrayFile &amp;&amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayFile</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a09d809f27869d6bf9bace2efdaabd415</anchor>
      <arglist>(size_t dimension, MPI_Comm comm=comm_global(), const std::string &amp;directory=&quot;.&quot;)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayFile</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a26300675242f1aa928b662603ec1a4b0</anchor>
      <arglist>(std::unique_ptr&lt; Distribution &gt; distribution, MPI_Comm comm=comm_global(), const std::string &amp;directory=&quot;.&quot;)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayFile</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a1214968f396e7e481f80d50ae81faa87</anchor>
      <arglist>(const DistrArray &amp;source)</arglist>
    </member>
    <member kind="function">
      <type>DistrArrayFile &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>ae036e349ee02fc7e36a966c8d14d18e3</anchor>
      <arglist>(const DistrArrayFile &amp;source)=delete</arglist>
    </member>
    <member kind="function">
      <type>DistrArrayFile &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>afdd1e614d088f441008a7e83c4e4d44c</anchor>
      <arglist>(DistrArrayFile &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~DistrArrayFile</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>afc4e85523e2d5a3ea1353bacbe56410d</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>compatible</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a24cf11319b198df2238cab728fe81a29</anchor>
      <arglist>(const DistrArrayFile &amp;source) const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>erase</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a0e0a9bf6423f365790cdc4b8d71739ef</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>at</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a869cbf0f14a83262bd5b5e137bfe44a9</anchor>
      <arglist>(index_type ind) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a4bf53e2e0509ad1e905cceecbd004f0f</anchor>
      <arglist>(index_type ind, value_type val) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a711dc83abc360ef2fa7c34f82968d9a9</anchor>
      <arglist>(index_type lo, index_type hi, value_type *buf) const override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>ac9faf610e6ac8b882c798c8e558a9e2a</anchor>
      <arglist>(index_type lo, index_type hi) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>put</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a069538c06860abbe214f53ef5faa3c3b</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a707fb73cca3027e502024d511e689ac5</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>gather</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a9b888dee28aefae63d83f7834efc3092</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scatter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a3d3d3544404ab721740313726b7395ad</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scatter_acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a62588bf097466a32e879cc8ab7920fbc</anchor>
      <arglist>(std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>vec</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>a12c669a0ee4b8117a03916bf22a5f3e7</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static DistrArrayFile</type>
      <name>CreateTempCopy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>ae1b0f75756ae323a48335df8ba3d9648</anchor>
      <arglist>(const DistrArray &amp;source, const std::string &amp;directory=&quot;.&quot;)</arglist>
    </member>
    <member kind="friend">
      <type>friend void</type>
      <name>swap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayFile.html</anchorfile>
      <anchor>ac98d0d2973b9ccdd2ad7b00606a631c1</anchor>
      <arglist>(DistrArrayFile &amp;x, DistrArrayFile &amp;y) noexcept</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::DistrArrayGA</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</filename>
    <base>molpro::linalg::array::DistrArray</base>
    <class kind="struct">molpro::linalg::array::DistrArrayGA::LocalBufferGA</class>
    <member kind="function">
      <type></type>
      <name>DistrArrayGA</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a9a0b279776d300ccfa1c8f76183c5c79</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayGA</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a55d88a467e07e596ee0c6ccf14df169e</anchor>
      <arglist>(size_t dimension, MPI_Comm commun)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayGA</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>af29c9c0e0270ac5488ac976cd9efceaa</anchor>
      <arglist>(const DistrArrayGA &amp;source)</arglist>
    </member>
    <member kind="function">
      <type>DistrArrayGA &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a699bbe26f1dcf50ca5dae42b13e35cb4</anchor>
      <arglist>(const DistrArrayGA &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayGA</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a98b7eb867b80073091e859a8556e4704</anchor>
      <arglist>(DistrArrayGA &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type>DistrArrayGA &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>ad6387fe48d839e8f12ad270b12112f74</anchor>
      <arglist>(DistrArrayGA &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~DistrArrayGA</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a23cc120c4831b83839448ccac33115f5</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>sync</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>aec67933877e0ca5adf437305e2ffa5c4</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const Distribution &amp;</type>
      <name>distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a8539ab3283e475207deab72dff77a6ef</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a4e0a346daa4e3965ad514529143fba58</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; const LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>aeecd84456912b7a8d4b3dfe593f47093</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>at</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a41776de7125451b69d5313c1dd3c99a4</anchor>
      <arglist>(index_type ind) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a2925ac7f26cb5451c0db2c1d813c835c</anchor>
      <arglist>(index_type ind, value_type val) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a8ea4d47f7fea173f44bf3b3e5eeb43da</anchor>
      <arglist>(index_type lo, index_type hi, value_type *buf) const override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>aa62e16ad1b8b7fb61f272616388d002e</anchor>
      <arglist>(index_type lo, index_type hi) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>put</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>ac52abab93526954cc98a40501c4bb3ee</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>ad3a1fac0d4e3640180e4b9cfc1361b60</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>gather</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>af9bec883bb3e3e4402e335020eb5c3a6</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scatter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>ae88d57c4335a490d2b315cb5c20bb35e</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scatter_acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a24f2a1bd89f0f0dc099a984c7a8586c7</anchor>
      <arglist>(std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>vec</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a062f309a6e121b1de0ed13862d549ad1</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>error</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a121a9861cbeb4baacc8e982b74ac16aa</anchor>
      <arglist>(const std::string &amp;message) const override</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>check_ga_ind_overlow</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a7c279563b90b65872c3529a1eaef3c63</anchor>
      <arglist>(index_type ind) const</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>Distribution</type>
      <name>make_distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a22a1bb7218bb901ab8b8c730f7f99a71</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>allocate_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a4d3b5ce9f5233eb10f11281766366713</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_comm_rank</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>ac942a569e38ba8e8f6f1790b2acf0b1c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_comm_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a8ca1b1175d6325e06526fdebb3b26d98</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_ga_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>adadf1ab481e051c50ed3b9d946c98a46</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_ga_pgroup</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a77c498296799aec5e8c01b81dd76fce5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_ga_chunk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a1ecab531628f81f8330e176b11c21fd8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_ga_allocated</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a278dae41e520ab10871a790409d92e2e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::unique_ptr&lt; Distribution &gt;</type>
      <name>m_distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a821113079671dbff05909d95a358e35a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected" static="yes">
      <type>static std::map&lt; MPI_Comm, int &gt;</type>
      <name>_ga_pgroups</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a125b1f34e8bf01edd8b5b93e22592f2c</anchor>
      <arglist></arglist>
    </member>
    <member kind="friend">
      <type>friend void</type>
      <name>swap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayGA.html</anchorfile>
      <anchor>a488bfdce763604fa19334cf819607b8e</anchor>
      <arglist>(DistrArrayGA &amp;a1, DistrArrayGA &amp;a2) noexcept</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::DistrArrayMPI3</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</filename>
    <base>molpro::linalg::array::DistrArray</base>
    <class kind="struct">molpro::linalg::array::DistrArrayMPI3::LocalBufferMPI3</class>
    <member kind="function">
      <type></type>
      <name>DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a880c788519ccc5edd3fcf6508e99523f</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>aa26a28fd94db789095eb914baa4f5365</anchor>
      <arglist>(size_t dimension, MPI_Comm commun)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>affed17163f91abb0ed39190c8961072d</anchor>
      <arglist>(std::unique_ptr&lt; Distribution &gt; distribution, MPI_Comm commun)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a518eade277039859ec760566e96b45f9</anchor>
      <arglist>(std::unique_ptr&lt; Distribution &gt; distribution, MPI_Comm commun, Span&lt; value_type &gt; buffer)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a9d0184264e90cd661d127acf7aa36db8</anchor>
      <arglist>(size_t dimension, MPI_Comm commun, Span&lt; value_type &gt; buffer)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a0ca41fe02f22ea6eea1348406020b61f</anchor>
      <arglist>(const DistrArrayMPI3 &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a5ab6d65062691084dfd08a08985dac1f</anchor>
      <arglist>(const DistrArray &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>adaf6b46440a824668c9e4f936ca0d624</anchor>
      <arglist>(DistrArrayMPI3 &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type>DistrArrayMPI3 &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>aec8b9c184e840a28a8db3d59b4e8c4b1</anchor>
      <arglist>(const DistrArrayMPI3 &amp;source)</arglist>
    </member>
    <member kind="function">
      <type>DistrArrayMPI3 &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a79f89a653d131906b5533d26572c31c1</anchor>
      <arglist>(DistrArrayMPI3 &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~DistrArrayMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a9c33c96152c5b1e54be688dfbd672d3a</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>sync</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a70187374c9fc893a5dde5a887a5da100</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const Distribution &amp;</type>
      <name>distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a5af2cf0f9dec48de361c8b69eda78fcd</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a1cc2152a8b18660431cb44b42e12f8c5</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; const LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a111164175ed99dfc16a73d85c59c95fb</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>at</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a50c6d83bd18e930b0f9d47394c763eff</anchor>
      <arglist>(index_type ind) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a34cff2e5ce2155d89303d41ce5757fe9</anchor>
      <arglist>(index_type ind, value_type val) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a6bf6a9b12d9df61fce13eb7dbb0e0808</anchor>
      <arglist>(index_type lo, index_type hi, value_type *buf) const override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a9463179c7646adbc969014fa06ecf9ff</anchor>
      <arglist>(index_type lo, index_type hi) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>put</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a3d8c675731145c06c49ccccf74ba3331</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a5083481028385815c875c68789d2bac0</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>gather</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a3d1b6677760d5ec2d5c0357ce245a571</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scatter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a059cb5db06277ba932d314864951d95c</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scatter_acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>ac7471b17c7b4da210c9f614996901a53</anchor>
      <arglist>(std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>vec</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a9bb945b979cf395bedb717404eae3c77</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>error</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a6dbee6c3038cc0a602d6a195f570a0ca</anchor>
      <arglist>(const std::string &amp;message) const override</arglist>
    </member>
    <member kind="enumeration" protection="protected">
      <type></type>
      <name>RMAType</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a43e14b96f07b56ed6865773c6c5fdcdb</anchor>
      <arglist></arglist>
      <enumvalue file="classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html" anchor="a43e14b96f07b56ed6865773c6c5fdcdbab5eda0a74558a342cf659187f06f746f">get</enumvalue>
      <enumvalue file="classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html" anchor="a43e14b96f07b56ed6865773c6c5fdcdba8e13ffc9fd9d6a6761231a764bdf106b">put</enumvalue>
      <enumvalue file="classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html" anchor="a43e14b96f07b56ed6865773c6c5fdcdba1673448ee7064c989d02579c534f6b66">acc</enumvalue>
      <enumvalue file="classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html" anchor="a43e14b96f07b56ed6865773c6c5fdcdba76a79a1cdd981d7b73a1b7cf9cc6e0de">gather</enumvalue>
      <enumvalue file="classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html" anchor="a43e14b96f07b56ed6865773c6c5fdcdba50894a237d9bcde0a18769af9a768baf">scatter</enumvalue>
      <enumvalue file="classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html" anchor="a43e14b96f07b56ed6865773c6c5fdcdba46fd877c999578e7b806b8c8136fe282">scatter_acc</enumvalue>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>_get_put</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a39c52221e0cb67cc37c4a50bfc63bd37</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *buf, RMAType option)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>_gather_scatter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a4e541a57a42c7dd67d19be154166bfa3</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices, std::vector&lt; value_type &gt; &amp;data, RMAType option)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>allocate_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a963adf4d4364dfbade5a8ee1cb551bb3</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>MPI_Win</type>
      <name>m_win</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>abc444902f7bc4347d976315b9a05be1b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::unique_ptr&lt; Distribution &gt;</type>
      <name>m_distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>adaecde0a26f290298ea0599bba662a9b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_allocated</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>a6228b3db743803420676ef19d2e9594a</anchor>
      <arglist></arglist>
    </member>
    <member kind="friend">
      <type>friend void</type>
      <name>swap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3.html</anchorfile>
      <anchor>ab1a73ebeb7c59c5c63e85f87f532f216</anchor>
      <arglist>(DistrArrayMPI3 &amp;a1, DistrArrayMPI3 &amp;a2) noexcept</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::DistrArraySpan</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</filename>
    <base>molpro::linalg::array::DistrArray</base>
    <class kind="struct">molpro::linalg::array::DistrArraySpan::LocalBufferSpan</class>
    <member kind="function">
      <type></type>
      <name>DistrArraySpan</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>ae9ab7ac9be1d87ff15e2f681c71adc89</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArraySpan</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a38f6789633cb1565a3d01c8ebba422a0</anchor>
      <arglist>(size_t dimension, Span&lt; value_type &gt; buffer, MPI_Comm commun=comm_global())</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArraySpan</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>ac715284f205f30461651fe08eb8dd525</anchor>
      <arglist>(std::unique_ptr&lt; Distribution &gt; distribution, Span&lt; value_type &gt; buffer, MPI_Comm commun=comm_global())</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArraySpan</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>aff1814e7123ba3d209256cda10733c11</anchor>
      <arglist>(const DistrArraySpan &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArraySpan</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a06a45bdfbd0722e5f15905da3879a9f3</anchor>
      <arglist>(const DistrArray &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrArraySpan</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>ad5359658f7e32afdfe8368a1fcc66d9f</anchor>
      <arglist>(DistrArraySpan &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type>DistrArraySpan &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a3f8f7892e84198d42191a649a643fb83</anchor>
      <arglist>(const DistrArraySpan &amp;source)</arglist>
    </member>
    <member kind="function">
      <type>DistrArraySpan &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>aa5aee47e1ebc939bb73439de674b41fc</anchor>
      <arglist>(DistrArraySpan &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>allocate_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>af765d3a1c8ade37c3f764c3528fb8972</anchor>
      <arglist>(Span&lt; value_type &gt; buffer)</arglist>
    </member>
    <member kind="function">
      <type>const Distribution &amp;</type>
      <name>distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a25a607e6713792f3174cad3daa8f3293</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>aafd23ecba850a9e258dce5b26eeed4a4</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; const LocalBuffer &gt;</type>
      <name>local_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>adb0db334f21e5bca311580c917e5eb4c</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>value_type</type>
      <name>at</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a504372cacb52c035bef86c1850e9cc15</anchor>
      <arglist>(index_type ind) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a1242a7b2393a49e18b3cc6a189ce55a1</anchor>
      <arglist>(index_type ind, value_type val) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a7cb1da387feb01e07b7fc0f3e1b5277d</anchor>
      <arglist>(index_type lo, index_type hi, value_type *buf) const override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a98924eab1730c42e7f8ff70be2544b0c</anchor>
      <arglist>(index_type lo, index_type hi) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>put</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>ab0d57d7cbadbb57092498e248ee9e684</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a46622ccd228a4b74b640379d64460024</anchor>
      <arglist>(index_type lo, index_type hi, const value_type *data) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>gather</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a195e0892b30b2b1742f1b500db85b932</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scatter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a3dabea87434bde2a3e254578a334ed1e</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>scatter_acc</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a9fd4a6cb14ff204d829a155e6c1d2f3b</anchor>
      <arglist>(std::vector&lt; index_type &gt; &amp;indices, const std::vector&lt; value_type &gt; &amp;data) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; value_type &gt;</type>
      <name>vec</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a70b0d17f39b7a0752eef55d6864fec91</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::unique_ptr&lt; Distribution &gt;</type>
      <name>m_distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a332eea51058b0e9711f39f2a6597c5e2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_allocated</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a00ce9e209b2f0e33a1432b70ed2a6846</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Span&lt; value_type &gt;</type>
      <name>m_span</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a0bdd9631ab7176af66f7e203674997cd</anchor>
      <arglist></arglist>
    </member>
    <member kind="friend">
      <type>friend void</type>
      <name>swap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArraySpan.html</anchorfile>
      <anchor>a1b03d36bd8e720c712c6113a6fb78fed</anchor>
      <arglist>(DistrArraySpan &amp;a1, DistrArraySpan &amp;a2) noexcept</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::util::DistrFlags</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</filename>
    <class kind="class">molpro::linalg::array::util::DistrFlags::Proxy</class>
    <member kind="function">
      <type></type>
      <name>DistrFlags</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a4e183517f6a844a4a457680174dfb5d4</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrFlags</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a9710c226e6ca7f65a25e75ef518191e3</anchor>
      <arglist>(const DistrFlags &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrFlags</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>aef8cc0a9f0ccfd75acd2f63fddf46701</anchor>
      <arglist>(DistrFlags &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type>DistrFlags &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>aeebc7117aa6e8097c314bec3ca33bcc6</anchor>
      <arglist>(const DistrFlags &amp;source)</arglist>
    </member>
    <member kind="function">
      <type>DistrFlags &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a6d651839c7f627de8e73ed9419dbe78a</anchor>
      <arglist>(DistrFlags &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~DistrFlags</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>accbd86155e2aa04772431fab3ec9eba2</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DistrFlags</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a12151540a614a07226f2f366fb2eebdd</anchor>
      <arglist>(MPI_Comm comm, int value=0)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>empty</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>abd47ac1000118b2102840ab1c4053369</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Proxy</type>
      <name>access</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a13fccab663f22b7db084437533404099</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Proxy</type>
      <name>access</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>af485cf0a2911a34a20af00cd8079ab49</anchor>
      <arglist>(int rank) const</arglist>
    </member>
    <member kind="function">
      <type>MPI_Comm</type>
      <name>communicator</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>acee0b7d40542b41b76b682dc60c34fc5</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>MPI_Comm</type>
      <name>m_comm</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a7e88d3b53526ea8d9884040c45a499a8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>MPI_Win</type>
      <name>m_win</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a0de5d27bd6d413ffaa144a3f8c26c655</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; int &gt;</type>
      <name>m_counter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a358e561aa75b337399aa09a08ff8f48e</anchor>
      <arglist></arglist>
    </member>
    <member kind="friend">
      <type>friend void</type>
      <name>swap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags.html</anchorfile>
      <anchor>a3d2b1ecc10049294cab629e8c6a09530</anchor>
      <arglist>(DistrFlags &amp;x, DistrFlags &amp;y)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::util::Distribution</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</filename>
    <templarg>typename Ind</templarg>
    <member kind="typedef">
      <type>Ind</type>
      <name>index_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a768002ff08c39ed609546b90cf9183e3</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a3f803984654c06c80b302bdd89fded70</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a94e0c9e0dbdadbb794157c1df878676d</anchor>
      <arglist>(const Distribution&lt; Ind &gt; &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a7f12bcecc04fe793cac5aed9682b5efe</anchor>
      <arglist>(Distribution&lt; Ind &gt; &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>Distribution&lt; Ind &gt; &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a1788779e64dcc786504e49ea45baa2f4</anchor>
      <arglist>(const Distribution&lt; Ind &gt; &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>Distribution&lt; Ind &gt; &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>ae9d511a117552cabef9200fbd7b0df2b</anchor>
      <arglist>(Distribution&lt; Ind &gt; &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~Distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a4c79935cf69f1af0173ce84e64df8db9</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Distribution</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>aaa0d2c8660889ec133ca3b6d0f9f427f</anchor>
      <arglist>(const std::vector&lt; index_type &gt; &amp;indices)</arglist>
    </member>
    <member kind="function">
      <type>std::pair&lt; int, int &gt;</type>
      <name>cover</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a455eb9c9c15ab02b3c2460a8256e4adb</anchor>
      <arglist>(index_type lo, index_type hi) const</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>cover</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>aaab77a45b0384c0e69036ca72446686e</anchor>
      <arglist>(index_type ind) const</arglist>
    </member>
    <member kind="function">
      <type>std::pair&lt; index_type, index_type &gt;</type>
      <name>range</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>af037933af27f330b29717cdadb267dcf</anchor>
      <arglist>(int chunk) const</arglist>
    </member>
    <member kind="function">
      <type>std::pair&lt; index_type, index_type &gt;</type>
      <name>border</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a389ce92ae9a0c1cfb045f8484a609bd0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a457dc89148f85b831bc603f2aaf6aa58</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; index_type &gt; &amp;</type>
      <name>chunk_borders</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a03fe779a8e5239c0977b6b4b9bde1b67</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>compatible</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a5e4c2e7205d9e73ae3be3be975a1716d</anchor>
      <arglist>(const Distribution&lt; Ind &gt; &amp;other) const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; index_type &gt;</type>
      <name>m_chunk_borders</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>af1930b1026bbf2f0fb692bb4cf8adb78</anchor>
      <arglist></arglist>
    </member>
    <member kind="friend">
      <type>friend void</type>
      <name>swap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Distribution.html</anchorfile>
      <anchor>a2c2638e10da1aa53ae75a8775eedb80a</anchor>
      <arglist>(Distribution&lt; index_type &gt; &amp;l, Distribution&lt; index_type &gt; &amp;r)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::DSpace</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</filename>
    <templarg>class Qt</templarg>
    <member kind="typedef">
      <type>Qt</type>
      <name>Q</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a1ed7ca80ffada004f1e78543dd76780c</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; Q, Q &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a7f5f883d26c6f6049ab7c066fc8f9378</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; Q, Q &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>aa45b77bea8e08619691919167ada303b</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DSpace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a6f5ebe6ae19a76ed2bb3c6d2fb0c6716</anchor>
      <arglist>(std::shared_ptr&lt; Logger &gt; logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>update</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>ad19d40d23ad3703547016071e4aa5c35</anchor>
      <arglist>(VecRef&lt; Q &gt; &amp;params, VecRef&lt; Q &gt; &amp;actions)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>clear</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a4af9a676ad8dcbc9aa1794ae711938dc</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>erase</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>ab2d0f4616ea1f8746de71a8040de8de1</anchor>
      <arglist>(size_t i)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a9ccb71e2aa6afca0ef7c9b1719a70bef</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; Q &gt;</type>
      <name>params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a71ea048ac9313c9a8c6501bbe5aaaebd</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>ac539ec628352901dedc593b0a814ba3a</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>cparams</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a0d720d3670bb27864d8994380134ce5d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; Q &gt;</type>
      <name>actions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a99a2488d44ef4ef035cdc94d0a868b4d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>actions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a68d96f1cec52cab3d2bd31be123480a5</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>cactions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a36dec1f7d98549d367d13f24178ce540</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>m_logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a244f49065009f584e05eee127d498a44</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; Q &gt;</type>
      <name>m_params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a2f7d03bcc546d57beb2f3d6c9b325c3c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; Q &gt;</type>
      <name>m_actions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1DSpace.html</anchorfile>
      <anchor>a60bbd0ac2db6b46c3b825bf070bffffc</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::detail::DSpaceResetter</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</filename>
    <templarg>class Q</templarg>
    <member kind="function">
      <type></type>
      <name>DSpaceResetter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>ac4efef2ba4e3ef8a6f64dec34a1f7fcf</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DSpaceResetter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>afaa739aa1f6ffb031c2be25c611ef109</anchor>
      <arglist>(int nreset, int max_Qsize)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>do_reset</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>a9d1181f8c9d6d03cb504b1f72af063a2</anchor>
      <arglist>(size_t iter, const subspace::Dimensions &amp;dims)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_nreset</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>ae19f0aaa545096962daa3379661d77c2</anchor>
      <arglist>(size_t i)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>get_nreset</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>a27b03a2d4922c913b5c7725b2fe8b8e3</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_max_Qsize</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>abbbd83bf92d578579c5cb75dcda5d1ef</anchor>
      <arglist>(size_t i)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>get_max_Qsize</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>aebb756f173cea19c6e499f75b695c98f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; int &gt;</type>
      <name>run</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>acef5af805bc31f9d4cd8fb6a8191f0d0</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;rparams, subspace::IXSpace&lt; R, Q, P &gt; &amp;xspace, const subspace::Matrix&lt; value_type &gt; &amp;solutions, const value_type_abs norm_thresh, const value_type_abs svd_thresh, ArrayHandlers&lt; R, Q, P &gt; &amp;handlers, Logger &amp;logger)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_nreset</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>af9bdc742d7628ba3eddbefbac2e75a45</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_max_Qsize_after_reset</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>a4dcb1c408145f9dc7d6494afefa7eeed</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::list&lt; Q &gt;</type>
      <name>solution_params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1detail_1_1DSpaceResetter.html</anchorfile>
      <anchor>aa99c75dd44f4e6947199bb2bd1c8decf</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::has_iterator</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1has__iterator.html</filename>
    <templarg>typename T</templarg>
    <member kind="function" static="yes">
      <type>static constexpr std::true_type</type>
      <name>test</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1has__iterator.html</anchorfile>
      <anchor>aa331b1b46b56fccb7cb6dc063ec04114</anchor>
      <arglist>(typename C::iterator *)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr std::false_type</type>
      <name>test</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1has__iterator.html</anchorfile>
      <anchor>ac0adf9860cb4319b830e5c2f3013eb78</anchor>
      <arglist>(...)</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr bool</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1has__iterator.html</anchorfile>
      <anchor>a8aeca0f6d3a1daa19bc543adade17ccd</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::has_mapped_type</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1has__mapped__type.html</filename>
    <templarg>class A</templarg>
    <templarg>class</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::has_mapped_type&lt; A, void_t&lt; typename A::mapped_type &gt; &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1has__mapped__type_3_01A_00_01void__t_3_01typename_01A_1_1mapped__type_01_4_01_4.html</filename>
    <templarg>class A</templarg>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::Interpolate</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1Interpolate.html</filename>
    <class kind="struct">molpro::linalg::itsolv::Interpolate::point</class>
    <member kind="function">
      <type></type>
      <name>Interpolate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Interpolate.html</anchorfile>
      <anchor>ab833e4c02392a88c0cfdd9129b964fe0</anchor>
      <arglist>(point p0, point p1, std::string interpolant=&quot;cubic&quot;, int verbosity=0)</arglist>
    </member>
    <member kind="function">
      <type>point</type>
      <name>operator()</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Interpolate.html</anchorfile>
      <anchor>aa9fe4e3831c40accb585c871627a0838</anchor>
      <arglist>(double x) const</arglist>
    </member>
    <member kind="function">
      <type>Interpolate::point</type>
      <name>minimize</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Interpolate.html</anchorfile>
      <anchor>affbc076b85dae30880c9f0aca5f72354</anchor>
      <arglist>(double xa, double xb, size_t bracket_grid=100, size_t max_bracket_grid=100000, bool analytic=true) const</arglist>
    </member>
    <member kind="function">
      <type>Interpolate::point</type>
      <name>minimize_cubic</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Interpolate.html</anchorfile>
      <anchor>ae3403a7c5ea1be1d34b51549b29737aa</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; double &gt; &amp;</type>
      <name>parameters</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Interpolate.html</anchorfile>
      <anchor>a489d1024ad31db058d7c12a7ccac230f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static std::vector&lt; std::string &gt;</type>
      <name>interpolants</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Interpolate.html</anchorfile>
      <anchor>a3568bfb5d55a1de2b4aeb21b67ce6646</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="friend">
      <type>friend std::ostream &amp;</type>
      <name>operator&lt;&lt;</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Interpolate.html</anchorfile>
      <anchor>a99e796dd3705718bb375477d04e4fa28</anchor>
      <arglist>(std::ostream &amp;os, const Interpolate &amp;interpolant)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::is_complex</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1is__complex.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::is_complex&lt; std::complex&lt; T &gt; &gt;</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1is__complex_3_01std_1_1complex_3_01T_01_4_01_4.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::is_disk</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1is__disk.html</filename>
    <templarg>class T</templarg>
    <templarg>typename</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::is_disk&lt; T, void_t&lt; typename T::disk_array &gt; &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1is__disk_3_01T_00_01void__t_3_01typename_01T_1_1disk__array_01_4_01_4.html</filename>
    <templarg>class T</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::is_distributed</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1is__distributed.html</filename>
    <templarg>class T</templarg>
    <templarg>typename</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::is_distributed&lt; T, void_t&lt; typename T::distributed_array &gt; &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1is__distributed_3_01T_00_01void__t_3_01typename_01T_1_1distributed__array_01_4_01_4.html</filename>
    <templarg>class T</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::is_iterable</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1is__iterable.html</filename>
    <templarg>class T</templarg>
    <templarg>typename</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::is_iterable&lt; T, void_t&lt; decltype(std::begin(std::declval&lt; T &gt;())), decltype(std::end(std::declval&lt; T &gt;())), std::enable_if_t&lt;!is_sparse_v&lt; T &gt; &gt; &gt; &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1is__iterable_3_01T_00_01void__t_3_01decltype_07std_1_1begin_0db4a62f7619457fd09db7752698d7624.html</filename>
    <templarg>class T</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::util::detail::is_one_of</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1is__one__of.html</filename>
    <templarg>typename T1</templarg>
    <templarg>typename T2</templarg>
    <templarg>typename... Ts</templarg>
    <member kind="variable" static="yes">
      <type>static constexpr bool</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1is__one__of.html</anchorfile>
      <anchor>a7fa483acdbb9358ef6f7376369c41a78</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::util::detail::is_one_of&lt; T1, T2 &gt;</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1is__one__of_3_01T1_00_01T2_01_4.html</filename>
    <templarg>typename T1</templarg>
    <templarg>typename T2</templarg>
    <member kind="variable" static="yes">
      <type>static constexpr bool</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1is__one__of_3_01T1_00_01T2_01_4.html</anchorfile>
      <anchor>a7656b5bb6250b6c641e39d69d1d6bbb8</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::is_std_array</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1is__std__array.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::is_std_array&lt; std::array&lt; T, N &gt; &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1is__std__array_3_01std_1_1array_3_01T_00_01N_01_4_01_4.html</filename>
    <templarg>typename T</templarg>
    <templarg>std::size_t N</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::ISubspaceSolver</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</filename>
    <templarg>class RT</templarg>
    <templarg>class QT</templarg>
    <templarg>class PT</templarg>
    <member kind="typedef">
      <type>RT</type>
      <name>R</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>aaf95ad005449309dfa69a7424c25d38a</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>QT</type>
      <name>Q</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>a96f96d22a10e2b6956a6e4e2356d6b29</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>PT</type>
      <name>P</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>a6473328a2313c2b24bc7cb238c486ef2</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; R, R &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>a2a06a811f18cef5088037cafa4eceedc</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; R, R &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>a2f17b16191359344bdbd73d20c6d66f9</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~ISubspaceSolver</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>ad7cfefb4b20a24f38099ae54e43d7fbe</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>solve</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>a35b9e44966f3b62ecdd402f330c5088b</anchor>
      <arglist>(IXSpace&lt; R, Q, P &gt; &amp;xspace, size_t nroots_max)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_error</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>a1a5b52ec31a14cc17015f51855c1121e</anchor>
      <arglist>(int root, value_type_abs error)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_error</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>aaa8b93d2137e355b08e490922572a359</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const std::vector&lt; value_type_abs &gt; &amp;errors)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const Matrix&lt; value_type &gt; &amp;</type>
      <name>solutions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>af48f67e3ba6cac80659e0bb169d79b99</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const std::vector&lt; value_type &gt; &amp;</type>
      <name>eigenvalues</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>a7b03cf4a324c80cbbe5c746e5f728891</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const std::vector&lt; value_type_abs &gt; &amp;</type>
      <name>errors</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>a2a9ce5a6199fa417a05d26ff517039a1</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual size_t</type>
      <name>size</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1ISubspaceSolver.html</anchorfile>
      <anchor>aac76eee68ef881c8f3e5da00321a9ce2</anchor>
      <arglist>() const =0</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::IterativeSolver</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <member kind="typedef">
      <type>typename R::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a027b41582a1fff9ebb679e6ae6e2fe5f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; R, Q &gt;::value_type</type>
      <name>scalar_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a4ce4d7778dab44edee1bea7075d7f227</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; R, R &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a17dcf15a66ef5c759d8a8a9a5616ce92</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::vector&lt; value_type &gt;</type>
      <name>VectorP</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a83cc580bb3f1dfab899fa9b4cd68babf</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::function&lt; void(const std::vector&lt; VectorP &gt; &amp;, const CVecRef&lt; P &gt; &amp;, const VecRef&lt; R &gt; &amp;)&gt;</type>
      <name>fapply_on_p_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a64287edfda9bc2abbdafd958b6cfe682</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~IterativeSolver</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>abc808a3db4d9f21e50b071fee35cb083</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>IterativeSolver</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a7e49705c8f5b805ff69f535ffcba5efc</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>IterativeSolver</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a0dab6f5601a84fa64d00e8e5628a3a1e</anchor>
      <arglist>(const IterativeSolver&lt; R, Q, P &gt; &amp;)=delete</arglist>
    </member>
    <member kind="function">
      <type>IterativeSolver&lt; R, Q, P &gt; &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>ac79eab8cc43e2ee4a4f1cdff0fb53625</anchor>
      <arglist>(const IterativeSolver&lt; R, Q, P &gt; &amp;)=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>IterativeSolver</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a3becd034ba0358533dda8362a1283a01</anchor>
      <arglist>(IterativeSolver&lt; R, Q, P &gt; &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>IterativeSolver&lt; R, Q, P &gt; &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a91e37bb38e22e0add6749afe16cc15c0</anchor>
      <arglist>(IterativeSolver&lt; R, Q, P &gt; &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual bool</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a527854d92a5f67f6bee487f46ea02524</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;actions, const Problem&lt; R &gt; &amp;problem, bool generate_initial_guess=false)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual bool</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a15457ee3cf4ffe270f869a3bedae6ef3</anchor>
      <arglist>(R &amp;parameters, R &amp;actions, const Problem&lt; R &gt; &amp;problem, bool generate_initial_guess=false)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual bool</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>ac87a7c8e687fb61521e51daed48b1ba5</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;actions, const Problem&lt; R &gt; &amp;problem, bool generate_initial_guess=false)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual int</type>
      <name>add_vector</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a76529da2bee144bb54981e0350702554</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;actions)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual int</type>
      <name>add_vector</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a9a83c4f944907a2a74f115d99062a073</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual int</type>
      <name>add_vector</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a2a35595f2fb582a71f2b9fbe7164af4a</anchor>
      <arglist>(R &amp;parameters, R &amp;action, value_type value=0)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual size_t</type>
      <name>add_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a9f2cb71b3bc51e8163c0658113fb2929</anchor>
      <arglist>(const CVecRef&lt; P &gt; &amp;pparams, const array::Span&lt; value_type &gt; &amp;pp_action_matrix, const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;action, fapply_on_p_type apply_p)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>clearP</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a2720fd380825d896d248c1a45a2f5b01</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>solution</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a520b858fe56b45bf13090a62854f2586</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;residual)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>solution_params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a597a254411508028bc9638b59ab9b639</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const VecRef&lt; R &gt; &amp;parameters)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>acfce319e750a3e3a925c489b91dbf7e2</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;residual)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual bool</type>
      <name>end_iteration_needed</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>acc90c71218893490dbf3ef515c1a9b14</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::vector&lt; size_t &gt;</type>
      <name>suggest_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>afa96ff75d67276bc116c0c13eeb2de29</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;solution, const CVecRef&lt; R &gt; &amp;residual, size_t max_number, double threshold)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>solution</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>aa0abcaa62ea4fb7c66061e6cae6d4247</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;residual)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>solution</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>aa8fabc3bee2f9d6d589e239f8d0aedbf</anchor>
      <arglist>(R &amp;parameters, R &amp;residual)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>solution_params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>ae0725b38684c71dc582fa03220b5b685</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, std::vector&lt; R &gt; &amp;parameters)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>solution_params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a8262a59edc2354bb3e8055cfc4fbd8cd</anchor>
      <arglist>(R &amp;parameters)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>af866452579638ebfa72eb5aa4e79cb97</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a1e6b71a13a52437b18b22ace59fcc489</anchor>
      <arglist>(R &amp;parameters, R &amp;action)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const std::vector&lt; int &gt; &amp;</type>
      <name>working_set</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>afee9d49b3360468c417d31259c740b1b</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::vector&lt; scalar_type &gt;</type>
      <name>working_set_eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>acab3c3acb752c4c5d093cfe92923eda6</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual size_t</type>
      <name>n_roots</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>afdeae214ff9e4a8433409a8cd87b54e8</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_n_roots</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a53c71a0d711b5ca06bf7c7858532485d</anchor>
      <arglist>(size_t nroots)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const std::vector&lt; scalar_type &gt; &amp;</type>
      <name>errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>af7a498d8483214bb9fbbdaabcb469332</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const Statistics &amp;</type>
      <name>statistics</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>af3f63de0aab95e5c15d1d09f034c664d</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a328990c674d38121f8ebbd5b51a26af0</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a0a4cbea70759f428d0934481642532b0</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_convergence_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>af28ce838454c794908965639d9425b75</anchor>
      <arglist>(double thresh)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual double</type>
      <name>convergence_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>aae54969171361e0982c8336dbf087dc4</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_convergence_threshold_value</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>ae5674166e14c2db01d9c9dcac437831f</anchor>
      <arglist>(double thresh)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual double</type>
      <name>convergence_threshold_value</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a35298af04d9f0436ba2d41fd16d62cd2</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_verbosity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a21beaa60fd363e453ee9fae8e62b0823</anchor>
      <arglist>(Verbosity v)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_verbosity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a069c10af4c2e24f5b689747042c1a9fd</anchor>
      <arglist>(int v)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual Verbosity</type>
      <name>get_verbosity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>aaf07c86f31393212706518abe4a231cc</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_max_iter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a9d505916315180ef208ab2c2f68bf04d</anchor>
      <arglist>(int n)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual int</type>
      <name>get_max_iter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>ae39e8d02193ed8d5690723433f5be0b8</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_max_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a6bd08099658ce5548af1fb1670c7cbfc</anchor>
      <arglist>(int n)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual int</type>
      <name>get_max_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>adde8118f66c77abc9df316bc63cf0355</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_p_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a42a08eb4824d0037887dd95167249088</anchor>
      <arglist>(double thresh)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual double</type>
      <name>get_p_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a36c67d3ad35534b41b49fc7bf931ebd3</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const subspace::Dimensions &amp;</type>
      <name>dimensions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a52bd10f0e2bb7d0f62a05fd0957d8c93</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a7e4e385f3faccb93d44a0e1b070bfa74</anchor>
      <arglist>(const Options &amp;options)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::shared_ptr&lt; Options &gt;</type>
      <name>get_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>ad22826350864db069a44a3a6aa1d15ea</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual scalar_type</type>
      <name>value</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a1a1d0c58c9e41e405bfa1cfff8179e1e</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual bool</type>
      <name>nonlinear</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a66ac1625fde41f28ade6cc29f81da931</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_profiler</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a05a40f7a9c104db49a1057896f3cc457</anchor>
      <arglist>(molpro::profiler::Profiler &amp;profiler)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const std::shared_ptr&lt; molpro::profiler::Profiler &gt; &amp;</type>
      <name>profiler</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a9378ef23b2490ab5b69a05205c6f57ad</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual bool</type>
      <name>test_problem</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolver.html</anchorfile>
      <anchor>a25e573d21d3d8eea3aa66f5cb8606aff</anchor>
      <arglist>(const Problem&lt; R &gt; &amp;problem, R &amp;v0, R &amp;v1, int verbosity=0, double threshold=1e-5) const =0</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::IterativeSolverTemplate</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</filename>
    <templarg>template&lt; class, class, class &gt; class Solver</templarg>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <member kind="function">
      <type></type>
      <name>IterativeSolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>acd19351fb4f3e8e98507cf0ec6b0032a</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>IterativeSolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>aac708b53560e38d7928e84868f6e1091</anchor>
      <arglist>(const IterativeSolverTemplate&lt; Solver, R, Q, P &gt; &amp;)=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>IterativeSolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>afc08e3ccc737019e8a77796deea4b417</anchor>
      <arglist>(IterativeSolverTemplate&lt; Solver, R, Q, P &gt; &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>IterativeSolverTemplate&lt; Solver, R, Q, P &gt; &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a6f38f1278d9a9f6c3c170b188114ed8b</anchor>
      <arglist>(const IterativeSolverTemplate&lt; Solver, R, Q, P &gt; &amp;)=delete</arglist>
    </member>
    <member kind="function">
      <type>IterativeSolverTemplate&lt; Solver, R, Q, P &gt; &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a143dac00773f825e5f5221873a2ef351</anchor>
      <arglist>(IterativeSolverTemplate&lt; Solver, R, Q, P &gt; &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>add_vector</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>aa5ed89e3da36e7f9e4b065213826e703</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>add_vector</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a91aaa9e988aa4524bed5a639314b6b2c</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>add_vector</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ac96d822e308648c5e295389bcdd460ea</anchor>
      <arglist>(R &amp;parameters, R &amp;actions, value_type value=0) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>add_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a1d1c168044d4511885f06f8aa50f618e</anchor>
      <arglist>(const CVecRef&lt; P &gt; &amp;pparams, const array::Span&lt; value_type &gt; &amp;pp_action_matrix, const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;actions, fapply_on_p_type apply_p) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>clearP</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>aa72cab3838296b747e499d86682b3728</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solution</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>af4e0f137f12bf9731d347f9dcc451b0b</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;residual) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solution</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a493bdf1e89e15c55b9a82425864ca610</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;residual) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solution</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a97c813a5516113c4fdc5dc9abd44fa87</anchor>
      <arglist>(R &amp;parameters, R &amp;residual) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solution_params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a238c04489ae2e21f9f188a80aacbae11</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, std::vector&lt; R &gt; &amp;parameters) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solution_params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>aa292796dc4eed09b0131e887f0f5c174</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const VecRef&lt; R &gt; &amp;parameters) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solution_params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a92c5d693781ef363399bdc62832135f9</anchor>
      <arglist>(R &amp;parameters) override</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; size_t &gt;</type>
      <name>suggest_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>aba447f125747f4a70c8cabafc6c9143a</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;solution, const CVecRef&lt; R &gt; &amp;residual, size_t max_number, double threshold) override</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; int &gt; &amp;</type>
      <name>working_set</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>abeb52c0ee58811527f789e9675674cb0</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>n_roots</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a6d328cf80c92a542338fcaf30f8e0606</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_n_roots</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a326a2d0f672de203c54bbde4b7020368</anchor>
      <arglist>(size_t roots) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ac3edaefda50129b65bd9b829a3d7d931</anchor>
      <arglist>(const Options &amp;options) override</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; Options &gt;</type>
      <name>get_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a6fc49ccc780ab47d600a06ab45064636</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; scalar_type &gt; &amp;</type>
      <name>errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ab105433d45a262ddf9adbd4cb36e9628</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const Statistics &amp;</type>
      <name>statistics</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a93bf7e1af72600b6293d3a072eedeaa6</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a2586061786c83f40ee932189f6a59353</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ab4748964a82d882be6cfebd7c158511e</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_convergence_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a215df7cdb4fd2d07f3c7f4acb7c58f34</anchor>
      <arglist>(double thresh) override</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>convergence_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a6d42e5e1955b041a57e421a4439d53c0</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_convergence_threshold_value</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>aa13ca336fed0aea5f331e023f3d626b1</anchor>
      <arglist>(double thresh) override</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>convergence_threshold_value</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a76b475dca1b9e715b5aefd549acc48dc</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_verbosity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ad6ffddd7ef42abcddf96847b6fe3e897</anchor>
      <arglist>(Verbosity v) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_verbosity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>af1240a8405dbaa9cda64e3d684bce2d5</anchor>
      <arglist>(int v) override</arglist>
    </member>
    <member kind="function">
      <type>Verbosity</type>
      <name>get_verbosity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>af20d7396488162b24ab0565201132fc0</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_max_iter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a4d206ec8158a042fa45b50eb417dd24d</anchor>
      <arglist>(int n) override</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>get_max_iter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a227f5b950f220fef378ee86ec7b9da92</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_max_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a5f1240d8cc340f434400894f5504e913</anchor>
      <arglist>(int n) override</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>get_max_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a64f2b7857436d5df6cb76a1fa51f4135</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_p_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a8f49c4695490cbd7107eddda0a5bc7b3</anchor>
      <arglist>(double threshold) override</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>get_p_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a261ed5e2536b16ce500022eac143b541</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const subspace::Dimensions &amp;</type>
      <name>dimensions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a7a83306888d70445233f01d4020d111a</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>scalar_type</type>
      <name>value</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a2cfe4f0a54b136895de82f5a357e4cd4</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_profiler</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a17509a0c4ccbd30a313f3e606dc54696</anchor>
      <arglist>(molpro::profiler::Profiler &amp;profiler) override</arglist>
    </member>
    <member kind="function">
      <type>const std::shared_ptr&lt; molpro::profiler::Profiler &gt; &amp;</type>
      <name>profiler</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a7b76f421dedacaaaac6d4da01b80e3cd</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ad86a191289fae16f2d6df34f3b3ddad4</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;actions, const Problem&lt; R &gt; &amp;problem, bool generate_initial_guess=false) override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>aaa15c94f22aef32292f8c81bc4d5b4dc</anchor>
      <arglist>(R &amp;parameters, R &amp;actions, const Problem&lt; R &gt; &amp;problem, bool generate_initial_guess=false) override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ac0f9db2eec5cb8d9153fe5ab23db017f</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;actions, const Problem&lt; R &gt; &amp;problem, bool generate_initial_guess=false) override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>test_problem</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a5d2794b425152880052b48e64116f295</anchor>
      <arglist>(const Problem&lt; R &gt; &amp;problem, R &amp;v0, R &amp;v1, int verbosity, double threshold) const override</arglist>
    </member>
    <member kind="function" protection="protected">
      <type></type>
      <name>IterativeSolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a8ae64ca06a5a8ddf6099eead01f66c23</anchor>
      <arglist>(std::shared_ptr&lt; subspace::IXSpace&lt; R, Q, P &gt; &gt; xspace, std::shared_ptr&lt; subspace::ISubspaceSolver&lt; R, Q, P &gt; &gt; solver, std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; handlers, std::shared_ptr&lt; Statistics &gt; stats, std::shared_ptr&lt; Logger &gt; logger)</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual</type>
      <name>~IterativeSolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a8da70aeae03caac6b144251c0d402d4c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual void</type>
      <name>set_value_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>abf9774c4ebf5d2ac8f670236d417ceb6</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="pure">
      <type>virtual void</type>
      <name>construct_residual</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ac38e5b3cb441edb495c6084a72dbdac0</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const CVecRef&lt; R &gt; &amp;params, const VecRef&lt; R &gt; &amp;actions)=0</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual bool</type>
      <name>linearEigensystem</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a92e68e1074cbc61076df22c6dc8d787a</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>size_t</type>
      <name>solve_and_generate_working_set</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a7b1a5bfc4fb3d6876ab8535be73e69ce</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;action)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>check_consistent_number_of_roots_and_solutions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a54c5badcc785eede77893bfe2c8c8cb9</anchor>
      <arglist>(const std::vector&lt; TTT &gt; &amp;roots, const size_t nparams)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>bool</type>
      <name>end_iteration_needed</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>aaabdae7c67ee909b5e7ddfc40fb46e0e</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt;</type>
      <name>m_handlers</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a3a9fde47c295d0370ba3d65b91bc9b95</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; subspace::IXSpace&lt; R, Q, P &gt; &gt;</type>
      <name>m_xspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a2dc73bdfa7e204116edab2d44b69e7cd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; subspace::ISubspaceSolver&lt; R, Q, P &gt; &gt;</type>
      <name>m_subspace_solver</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a61e18e30ed35370994bb53003a5d4642</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; double &gt;</type>
      <name>m_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ac50fb7b6de5f3cfa26e703337060a0da</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; double &gt;</type>
      <name>m_value_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>af75da06fcb4b67cbed43a74142eebb56</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; int &gt;</type>
      <name>m_working_set</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a289ff256b61bc1746f7dd0fddd0b8166</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>size_t</type>
      <name>m_nroots</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a9b5c7aee955bf41e7ad1d7a1889ce6e0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_convergence_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ab88aa2f7710140556b2e670a54730784</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_convergence_threshold_value</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ae91c0ec6524d77c5a789f61995a01c43</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; Statistics &gt;</type>
      <name>m_stats</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ae24be2854a659d0b3c5a3fe8db922e5e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>m_logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>abee358ce5516717e73670d75f2eebc45</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_normalise_solution</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ab53e871487e088b6502f51f6ec2ccab3</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>fapply_on_p_type</type>
      <name>m_apply_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a3dd09e6fa94c79029c5b876ea509edb9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Verbosity</type>
      <name>m_verbosity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ac3e460814c7e8040c9a6fed3de9e3026</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_max_iter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>ae0d0a39af9d4914c52f317d51f6fd084</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>size_t</type>
      <name>m_max_p</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a75b4f79b443412af94b4e9dbd19254bb</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_p_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a9cd1557340c1bdd64cf98ffdd3e6287a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_end_iteration_needed</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1IterativeSolverTemplate.html</anchorfile>
      <anchor>a9ee1b0fd61900d12fcaa7ba7663c55e1</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::BufferManager::Iterator</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</filename>
    <member kind="typedef">
      <type>std::forward_iterator_tag</type>
      <name>iterator_category</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a06230c7716ffad96ca26bce1cae3579f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::ptrdiff_t</type>
      <name>difference_type</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a43308df88cca249fdf84c45a844e3770</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>Span&lt; typename T::value_type &gt;</type>
      <name>value_type</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a9112b86782257926b2cdc20269895908</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>Span&lt; typename T::value_type &gt; *</type>
      <name>pointer</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>aacebebf6dceff2f2aa6de463efe0eab2</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>const Span&lt; typename T::value_type &gt; &amp;</type>
      <name>reference</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>af5cafdf8da25bebb7fd82e8112e024b8</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Iterator</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a5e953e7f21a8dd27afe039fd1cfe0b26</anchor>
      <arglist>(BufferManager &amp;manager, bool begin=false, bool end=false)</arglist>
    </member>
    <member kind="function">
      <type>reference</type>
      <name>operator*</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a7cc510a10cc03be27c1f86f488063223</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>pointer</type>
      <name>operator-&gt;</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a271e3d0797aebcc8b9d18988fa8bb812</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>Iterator &amp;</type>
      <name>operator++</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a48eba3b2150ec0560bd9501e474ad0ab</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>Iterator</type>
      <name>operator++</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a7051005b8d21e423c34d14cf701989aa</anchor>
      <arglist>(int)</arglist>
    </member>
    <member kind="friend">
      <type>friend bool</type>
      <name>operator==</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>adf9a9358b475317bd816d7b47a0f5cd1</anchor>
      <arglist>(const Iterator &amp;a, const Iterator &amp;b)</arglist>
    </member>
    <member kind="friend">
      <type>friend bool</type>
      <name>operator!=</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1BufferManager_1_1Iterator.html</anchorfile>
      <anchor>a27aed9f3ea67a677964fa51de28a5824</anchor>
      <arglist>(const Iterator &amp;a, const Iterator &amp;b)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::IXSpace</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</filename>
    <templarg>class RT</templarg>
    <templarg>class QT</templarg>
    <templarg>class PT</templarg>
    <member kind="typedef">
      <type>RT</type>
      <name>R</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a540ac7d74ec919e172526e788f3c0f85</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>QT</type>
      <name>Q</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a9396cdc6e084fc7f1b5c27490a43b6b0</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>PT</type>
      <name>P</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a5a96c31c79c2c0981fae16714099ae55</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; R, R &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>aa60e8f6f63ff781d8877ea06a8bf116f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; R, R &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a7a0ffeaa945af785af196a50bb5ad916</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>IXSpace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>adb7f294b86875bfe67cbbe382cde821e</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~IXSpace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>af6113d77157dc1b531ed81d845e31e74</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a5f90338e4f03d9298641a4ad3f603522</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>erase</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>ae3383d3d746c3d634f426953daf53072</anchor>
      <arglist>(size_t i)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>eraseq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a7393a3955f4493dcc4d36650159a2078</anchor>
      <arglist>(size_t i)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>erasep</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a718476898896233a153a5ed8ed02518c</anchor>
      <arglist>(size_t i)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>erased</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a3572d0dc1c4874323e52b10c65e22273</anchor>
      <arglist>(size_t i)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>update_pspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a90ffba67ffb0262cf996e73d0cd95354</anchor>
      <arglist>(const CVecRef&lt; P &gt; &amp;params, const array::Span&lt; value_type &gt; &amp;pp_action_matrix)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>update_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>aea25a4672a5baaeba2fcb36ad9b09b2e</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;params, const CVecRef&lt; R &gt; &amp;actions)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>update_dspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>afb897ccb015c8d42dbe57a6f0c92ae62</anchor>
      <arglist>(VecRef&lt; Q &gt; &amp;params, VecRef&lt; Q &gt; &amp;actions)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual VecRef&lt; P &gt;</type>
      <name>paramsp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a8672c92eecff09900f7d0083482df8dc</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual VecRef&lt; Q &gt;</type>
      <name>paramsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a38a43598c37b739d9dad7406442fe153</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual VecRef&lt; Q &gt;</type>
      <name>actionsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a31506221cef98c561d7ed213824301af</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual VecRef&lt; Q &gt;</type>
      <name>paramsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a3870c3b02300ecfc10d4f831809d4251</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual VecRef&lt; Q &gt;</type>
      <name>actionsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>ad56400b6eddefb92cf7a48a969e2430f</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; P &gt;</type>
      <name>paramsp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a2a71dc8c4caedaa74276dfb58703c71a</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>paramsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a522cde7457f11fec16c7a57d8a9002f2</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>actionsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>adfe901012de93a40b1c9a0308223a0db</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>paramsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a7b34d1a05bb235576d634e71a3eb80de</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>actionsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>abee3c36eaf580d194564dd05c681686d</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; P &gt;</type>
      <name>cparamsp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a2fa2b57ecefe5b4f2cad2f26851a833a</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>cparamsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a1ab2e5551cfff3610f3c4809c3381905</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>cactionsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a8c2d7979c2aecd18ac9453fc9fa9f971</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>cparamsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a97ce5eb7856f267cd5d60057dc6839d3</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>cactionsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a5fae0c231ee964c379ffc710e8bf276c</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual const Dimensions &amp;</type>
      <name>dimensions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>add51951e8630f1389adc79e56bb7a633</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="variable">
      <type>SubspaceData</type>
      <name>data</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1IXSpace.html</anchorfile>
      <anchor>a62b5c65e6258f13b670057780bd5ea3a</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandler::LazyHandle</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</filename>
    <member kind="typedef">
      <type>ArrayHandler&lt; AL, AR &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a6a3cffeecdcf7f3136b6a27849936268</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::reference_wrapper&lt; T &gt;</type>
      <name>ref_wrap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a8095230db2e9ea0bc676664c1888fc3d</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LazyHandle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a648de93bf4d1802ca4aabbbf0abff761</anchor>
      <arglist>(ArrayHandler&lt; AL, AR &gt; &amp;handler)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~LazyHandle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a23a0db1962c8e9f0217ace8b384a9d1a</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a080b8c891ee0a335943d4dabdea0c72b</anchor>
      <arglist>(value_type alpha, const AR &amp;x, AL &amp;y)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a20dbb482f7200a82cc0212b595000d17</anchor>
      <arglist>(const AL &amp;x, const AR &amp;y, value_type &amp;out)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>eval</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a0ccd48ee6a0c2d874ee5fe395f188f1c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>invalidate</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a3b0ae2897bbda5ae8b074fa06e762b0e</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>invalid</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>adc2f8a1d13761b128c1e1ed6adc08ba2</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>error</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a37d2271aef475029d8ee6dbfbec32be8</anchor>
      <arglist>(std::string message)</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual bool</type>
      <name>register_op_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>af477f947fe6c798ef3d964aae63582e6</anchor>
      <arglist>(const std::string &amp;type)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>clear</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a0f15d7a19a377c8460a142f68fea2e79</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::set&lt; std::string &gt;</type>
      <name>m_op_types</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>ad836493ce12da6c4bd02d927aaaff5d7</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>util::OperationRegister&lt; value_type, ref_wrap&lt; const AR &gt;, ref_wrap&lt; AL &gt; &gt;</type>
      <name>m_axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>adaa97da1577894bcbae46700767188d9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>util::OperationRegister&lt; ref_wrap&lt; const AL &gt;, ref_wrap&lt; const AR &gt;, ref_wrap&lt; value_type &gt; &gt;</type>
      <name>m_dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>a45f59e92dce6d9b68af12131b82b4dc8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>ArrayHandler&lt; AL, AR &gt; &amp;</type>
      <name>m_handler</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>ac1d6fbcbcac61819fbf795dbfef5aeb8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_invalid</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1LazyHandle.html</anchorfile>
      <anchor>ad8f638facc5830ee0ccfe381c189da34</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::LinearEigensystem</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystem.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>molpro::linalg::itsolv::IterativeSolver</base>
    <member kind="function" virtualness="pure">
      <type>virtual std::vector&lt; scalar_type &gt;</type>
      <name>eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystem.html</anchorfile>
      <anchor>a6c74967a8caabeeb48d4e5b9a12f42c8</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystem.html</anchorfile>
      <anchor>a66e43e2aa5f24212eb3562324554ce4f</anchor>
      <arglist>(bool hermitian)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual bool</type>
      <name>get_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystem.html</anchorfile>
      <anchor>a1280ef097d90b4b1f1ed608db76f1071</anchor>
      <arglist>() const =0</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::LinearEigensystemDavidson</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>IterativeSolverTemplate&lt; LinearEigensystem, R, R, std::map&lt; size_t, typename R::value_type &gt; &gt;</base>
    <member kind="typedef">
      <type>IterativeSolverTemplate&lt; LinearEigensystem, R, Q, P &gt;</type>
      <name>SolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>ae0ec3d737bc94543b35dc4d0e3558470</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LinearEigensystemDavidson</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>ae1fcec228a6eaf28bab62636828e5483</anchor>
      <arglist>(const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;(), const std::shared_ptr&lt; Logger &gt; &amp;logger_=std::make_shared&lt; Logger &gt;())</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>nonlinear</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a262897f82aa5a3ed099646e05ab73372</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a22ed76de01c073008465614a389df5c3</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a5f3c0bb3e790c82ffebbfbe62d1d3a9c</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>aebcf854909acb92f11bb2ea33a596c64</anchor>
      <arglist>(R &amp;parameters, R &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>precondition</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a66a707d2bc8900f8b0478c881ef99cf4</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action) const</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; scalar_type &gt;</type>
      <name>eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a57e0b6fcfe3f0711b5c618f0660f2c3c</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::vector&lt; scalar_type &gt;</type>
      <name>working_set_eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>ad0cc61ce681e6c9f07a4a869d1caf33d</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_value_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a8358829492685415686f92b83d7962fc</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>ac264beb56a500ce1a90edd391ad16b87</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_reset_D</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>ab4e8f66aaaa54da8eb41e2a5733ece1d</anchor>
      <arglist>(size_t n)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>get_reset_D</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a925f70a6fb1a01ede8dd5a6dc0b7dedb</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_reset_D_maxQ_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a241caed8a5b1e346fc978729892f6260</anchor>
      <arglist>(size_t n)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>get_reset_D_maxQ_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a1dabbe4326a656e6cb8f9a1018ceca3d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>get_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a053621c2fdae8dcf9c4542ee97888ff7</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a5b363a8031dc1b31e0d70783428a5ab9</anchor>
      <arglist>(int n)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a30cfe987c737da469e7a86a8725586a4</anchor>
      <arglist>(bool hermitian) override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>get_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a3a0109d914fa9d1c4f40fdc54b835efa</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>aa46edad2cecdac0a051062435a7188f1</anchor>
      <arglist>(const Options &amp;options) override</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; Options &gt;</type>
      <name>get_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>aea95c0b3b85e393fac9d39c931463c03</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="variable">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a3928db7987c24bda78d24e2f18e6c1d5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>propose_rspace_norm_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>abc0bdc92047dafddad208549048eef54</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>propose_rspace_svd_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a3b96209972872ffece08f78a2c747c5a</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>construct_residual</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>accbb21b9dc3c611e7bddd25d0750f972</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const CVecRef&lt; R &gt; &amp;params, const VecRef&lt; R &gt; &amp;actions) override</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a482db55c4914b82cedcbbb3190c519f8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>detail::DSpaceResetter&lt; Q &gt;</type>
      <name>m_dspace_resetter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a54f58e57011e81a3649959d5bf900b59</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a9bf051b5bc298da49a9046af0feb0104</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; double &gt;</type>
      <name>m_last_values</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a26f7dd1a80e4fdda081a2353d3cabb9e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_resetting_in_progress</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidson.html</anchorfile>
      <anchor>a4f8b13cbdbd2dc723312d0d640c31917</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::LinearEigensystemDavidsonOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</filename>
    <base>molpro::linalg::itsolv::LinearEigensystemOptions</base>
    <member kind="function">
      <type></type>
      <name>LinearEigensystemDavidsonOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</anchorfile>
      <anchor>aef598d37855f5fd61b4bfded5de45cd8</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LinearEigensystemDavidsonOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</anchorfile>
      <anchor>a93f23fc3a0766404a5171037b65d35eb</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>reset_D</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</anchorfile>
      <anchor>a0e53552bc93ee59f12f4b4ee39ee7af6</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>reset_D_max_Q_size</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</anchorfile>
      <anchor>a6ba88cdc8e7725a25ed7534119666303</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>max_size_qspace</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</anchorfile>
      <anchor>aa70d4f86a39f178faf148d005c84f7e5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>norm_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</anchorfile>
      <anchor>aba4aedaad0f569c5f36c4c315db03b30</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>svd_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</anchorfile>
      <anchor>a3afdce34fafd036d3f5a13ef3f5f0b60</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; bool &gt;</type>
      <name>hermiticity</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemDavidsonOptions.html</anchorfile>
      <anchor>af2b7bf0db9fec526a2f0c46be9462f06</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::LinearEigensystemOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemOptions.html</filename>
    <base>molpro::linalg::itsolv::Options</base>
    <member kind="function">
      <type></type>
      <name>LinearEigensystemOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemOptions.html</anchorfile>
      <anchor>adf9c9b0d65f3477e70a4c638bd0f4d23</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LinearEigensystemOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemOptions.html</anchorfile>
      <anchor>a53f3e69592b027beede7245d484ecfc9</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::LinearEigensystemRSPT</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>IterativeSolverTemplate&lt; LinearEigensystem, R, R, std::map&lt; size_t, typename R::value_type &gt; &gt;</base>
    <member kind="typedef">
      <type>IterativeSolverTemplate&lt; LinearEigensystem, R, Q, P &gt;</type>
      <name>SolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a9706701d165b289d50d3c9790c257a6a</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LinearEigensystemRSPT</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>adae189b963b15dedaadb562ce93a46c8</anchor>
      <arglist>(const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers, const std::shared_ptr&lt; Logger &gt; &amp;logger_=std::make_shared&lt; Logger &gt;())</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>nonlinear</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a9133d8604c6a8cfe09acd4324214977c</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>linearEigensystem</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a41c49e8eeb9939b4c75b332055ecbaa7</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>aa481d05b3a9ed63afa2edd45db35414d</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a3e730f56ae36aa0980112b33bf949594</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>adac97ddb9bf7d21e541789a4277fd2e4</anchor>
      <arglist>(R &amp;parameters, R &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>precondition</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>ac053cd110d99cd935aa22be81bb836c0</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action) const</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; scalar_type &gt;</type>
      <name>eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a8bf7934d86cb16fd036e14239b0f3fc2</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::vector&lt; scalar_type &gt;</type>
      <name>working_set_eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>ab66f2816ddfbdcaa81f5987a449222da</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a76fbd6531f33235601206f1cdafc089a</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a066e58ea525e6af0b301f219c99ce8bc</anchor>
      <arglist>(bool hermitian) override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>get_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a364903f7d2778386f00377457bba6a7d</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a340cd13fa9e2d978e896595086001e52</anchor>
      <arglist>(const Options &amp;options) override</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; Options &gt;</type>
      <name>get_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a1038595f7fc1605192f20e866a1e6be6</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="variable">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>ae1ad76f5444d668ecca13b4c16abe3ee</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>propose_rspace_norm_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a5884a9bbbfc0101fd01fc528de91e544</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>propose_rspace_svd_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a48ea5dfd740161c89349f6bddeda3ea3</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>construct_residual</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a9cc8d325386023837f7b47694c516d93</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const CVecRef&lt; R &gt; &amp;params, const VecRef&lt; R &gt; &amp;actions) override</arglist>
    </member>
    <member kind="function" protection="protected" static="yes">
      <type>static std::string</type>
      <name>str</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>ab7fdd8d63e9974bd2b474c02ffa2ef1c</anchor>
      <arglist>(const std::vector&lt; double &gt; &amp;a)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; double &gt;</type>
      <name>m_rspt_values</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPT.html</anchorfile>
      <anchor>a88aa99b67080273a156f748f346ebe6f</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::LinearEigensystemRSPTOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPTOptions.html</filename>
    <base>molpro::linalg::itsolv::LinearEigensystemOptions</base>
    <member kind="function">
      <type></type>
      <name>LinearEigensystemRSPTOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPTOptions.html</anchorfile>
      <anchor>ad1788b0e32efb0ccda131656b9985e3c</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LinearEigensystemRSPTOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPTOptions.html</anchorfile>
      <anchor>a39f25c0fab0f72507146d9b88392c040</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>norm_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPTOptions.html</anchorfile>
      <anchor>a6d229497830ee01ad13a7c11a613cd2c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>svd_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEigensystemRSPTOptions.html</anchorfile>
      <anchor>a697c48e0048dc454270e89b954d8e5b0</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::LinearEquations</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquations.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>molpro::linalg::itsolv::IterativeSolver</base>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>add_equations</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquations.html</anchorfile>
      <anchor>a40af5c8755ec2e795bc5889c14e57b5a</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;rhs)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>add_equations</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquations.html</anchorfile>
      <anchor>a930281aef2ec4263cc581c6d3c1b9698</anchor>
      <arglist>(const std::vector&lt; R &gt; &amp;rhs)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>add_equations</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquations.html</anchorfile>
      <anchor>af55044caaf97a1ba3ad6ac781285ac8d</anchor>
      <arglist>(const R &amp;rhs)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual CVecRef&lt; Q &gt;</type>
      <name>rhs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquations.html</anchorfile>
      <anchor>ab65be129e9371d0095937fff8d2792bc</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>set_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquations.html</anchorfile>
      <anchor>ad679017754a0b1af86f6799df7a12ead</anchor>
      <arglist>(bool hermitian)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual bool</type>
      <name>get_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquations.html</anchorfile>
      <anchor>a95a5d2ce5046045b4993711f3b5a82c7</anchor>
      <arglist>() const =0</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::LinearEquationsDavidson</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>IterativeSolverTemplate&lt; LinearEquations, R, R, std::map&lt; size_t, typename R::value_type &gt; &gt;</base>
    <member kind="typedef">
      <type>IterativeSolverTemplate&lt; LinearEquations, R, Q, P &gt;</type>
      <name>SolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a2bce4af8274b02f9a9f97d09b3f7833a</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LinearEquationsDavidson</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a56310ff60c45992c5dc838b171138e2a</anchor>
      <arglist>(const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers, const std::shared_ptr&lt; Logger &gt; &amp;logger_=std::make_shared&lt; Logger &gt;())</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a070cbeddff5d6ca1b30055fa7b30f8ef</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;actions, const Problem&lt; R &gt; &amp;problem, bool generate_initial_guess=false) override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>nonlinear</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a6bf15f4b37f912dc22393997154cdca7</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a22f10085590a24aebefad82536a7c7a3</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a6ebb83d95a1f6130a1aa2847f53196c8</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a465dbaaf2cab15ca07f1d662de6675c9</anchor>
      <arglist>(R &amp;parameters, R &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>add_equations</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a4b3ac46bb1923f18acbc1c985d4af7a1</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;rhs) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>add_equations</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a5b0aa5eb730099d33bd9076a3bc5eaff</anchor>
      <arglist>(const R &amp;rhs) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>add_equations</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a1726e87cb076209df93e371ee02d578d</anchor>
      <arglist>(const std::vector&lt; R &gt; &amp;rhs) override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>rhs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a8e3f14d5cd05418e40e614a20ffc4053</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_norm_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a59fd7897a1d2b405623fbcbe31d3c7a3</anchor>
      <arglist>(double thresh)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>get_norm_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a2393848822b65c218f6d425e89bfd9e2</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_svd_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a211a4e48e4723b609af546dcc90c72e4</anchor>
      <arglist>(double thresh)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>get_svd_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>ad4c8f30e7653281e580044176b7579b0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_reset_D</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>aa18de07639b2a59f4aab0fa00b3336ed</anchor>
      <arglist>(size_t n)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>get_reset_D</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a02ef4c76e400651eef9805a20a3e2350</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_reset_D_maxQ_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>ad70a9b4f851fc510bdf189a107ab68e2</anchor>
      <arglist>(size_t n)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>get_reset_D_maxQ_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a64c70aa7cafca66228994805f913fa91</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>af8393b6f0edfedf8331fb2bd0f88a1ed</anchor>
      <arglist>(int n)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>get_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>aef031425d38100193d223466f7741cf8</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a1a13c94801544066f87353fb5ec6e692</anchor>
      <arglist>(bool hermitian) override</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>get_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>aad9d42bd1b1b22ac1b6b0957d00a6cdc</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_augmented_hessian</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a1cb407598df11ffef8877bb72a5815c9</anchor>
      <arglist>(const double parameter)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>get_augmented_hessian</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a8a7ff1023a3dbc00675d862a6417021f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a6f6b26e5b00d9674a0b4d81128125893</anchor>
      <arglist>(const Options &amp;options) override</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; Options &gt;</type>
      <name>get_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a4bb63422b55572e405b7f7399691639e</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>aad10a92985b4c8ca6119e00096410f19</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a2586061786c83f40ee932189f6a59353</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>ab4748964a82d882be6cfebd7c158511e</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="variable">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a07c643a999d2eb1e02fe312d120dc94e</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>construct_residual</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a85d020e1e7abe04d905564573885a93a</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const CVecRef&lt; R &gt; &amp;params, const VecRef&lt; R &gt; &amp;actions) override</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_norm_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a168f12c22e8776185aa8d003d867d064</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_svd_thresh</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a41b9acd15e5e3b0b93f1fbcd3b826c48</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a7d15c6e7572aacd3572efb85b21ed4fd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>detail::DSpaceResetter&lt; Q &gt;</type>
      <name>m_dspace_resetter</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a0472ab791f0ddb7074cec65c3dbbca0e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidson.html</anchorfile>
      <anchor>a7a816bb027c4409f79fbe2f616350a1e</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::LinearEquationsDavidsonOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</filename>
    <base>molpro::linalg::itsolv::LinearEquationsOptions</base>
    <member kind="function">
      <type></type>
      <name>LinearEquationsDavidsonOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a5f53f8d21aeac30074d3d45682597584</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LinearEquationsDavidsonOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a5680e30e9e9017c91a7529169ab27b3c</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>reset_D</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a784889dc74ddb5e060ba0eb1bd17639d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>reset_D_max_Q_size</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a7cdccd25b63d7fe1e9226707fb0eeb96</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>max_size_qspace</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a838b66d55b31aa54a0169efaf9ac4078</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>norm_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a9c8b17c0dae35fd92e056883c2f2f806</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>svd_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a85531ec998e6148927a2d4b7d79094b1</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; bool &gt;</type>
      <name>hermiticity</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a0e488fb88aeca271776e7fd9dbd548cb</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>augmented_hessian</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsDavidsonOptions.html</anchorfile>
      <anchor>a365add6c79883e05d271facaa8495b89</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::LinearEquationsOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsOptions.html</filename>
    <base>molpro::linalg::itsolv::Options</base>
    <member kind="function">
      <type></type>
      <name>LinearEquationsOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsOptions.html</anchorfile>
      <anchor>a799771640cef8bcaf4e2753a7c52b738</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LinearEquationsOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1LinearEquationsOptions.html</anchorfile>
      <anchor>a6983668c6b678048ff8d61215ce24f01</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::DistrArray::LocalBuffer</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1DistrArray_1_1LocalBuffer.html</filename>
    <base>Span&lt; value_type &gt;</base>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~LocalBuffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray_1_1LocalBuffer.html</anchorfile>
      <anchor>af2043876402e35844e28326a927e1885</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>compatible</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray_1_1LocalBuffer.html</anchorfile>
      <anchor>a0b4b81e98471147d50a68c6001a73fe4</anchor>
      <arglist>(const LocalBuffer &amp;other) const</arglist>
    </member>
    <member kind="function">
      <type>size_type</type>
      <name>start</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray_1_1LocalBuffer.html</anchorfile>
      <anchor>ab7cd7eb5f8e8cbaaa79054d92f7a603f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>size_type</type>
      <name>m_start</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray_1_1LocalBuffer.html</anchorfile>
      <anchor>a3b07639f83164a1d7879314fd179ec82</anchor>
      <arglist></arglist>
    </member>
    <member kind="friend">
      <type>friend void</type>
      <name>swap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArray_1_1LocalBuffer.html</anchorfile>
      <anchor>a4c7377ed3e20e7947faf84fccd356a3e</anchor>
      <arglist>(LocalBuffer &amp;, LocalBuffer &amp;)=delete</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::DistrArrayDisk::LocalBufferDisk</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk_1_1LocalBufferDisk.html</filename>
    <base>molpro::linalg::array::DistrArray::LocalBuffer</base>
    <member kind="function">
      <type></type>
      <name>LocalBufferDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk_1_1LocalBufferDisk.html</anchorfile>
      <anchor>a2fdb9ef53c5ce7294b4014c767b4b50b</anchor>
      <arglist>(DistrArrayDisk &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LocalBufferDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk_1_1LocalBufferDisk.html</anchorfile>
      <anchor>a7e60f72d7330c188da4ecf34c0d0417e</anchor>
      <arglist>(DistrArrayDisk &amp;source, const span::Span&lt; value_type &gt; &amp;buffer)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~LocalBufferDisk</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk_1_1LocalBufferDisk.html</anchorfile>
      <anchor>a560b28394248aa197f64338e72a54b47</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>do_dump</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk_1_1LocalBufferDisk.html</anchorfile>
      <anchor>a8f58babec5898232b01599677d797398</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; value_type &gt;</type>
      <name>m_snapshot_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk_1_1LocalBufferDisk.html</anchorfile>
      <anchor>a0d1c06de52d347d7a3ed26532b4e6b70</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>DistrArrayDisk &amp;</type>
      <name>m_source</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1DistrArrayDisk_1_1LocalBufferDisk.html</anchorfile>
      <anchor>adb5a02e1bdcc0eb944bb4b3c611148ea</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::DistrArrayGA::LocalBufferGA</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1DistrArrayGA_1_1LocalBufferGA.html</filename>
    <base>molpro::linalg::array::DistrArray::LocalBuffer</base>
    <member kind="function">
      <type></type>
      <name>LocalBufferGA</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1DistrArrayGA_1_1LocalBufferGA.html</anchorfile>
      <anchor>a7863496278549fd06b432e5b2b435d54</anchor>
      <arglist>(DistrArrayGA &amp;source)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::DistrArrayMPI3::LocalBufferMPI3</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3_1_1LocalBufferMPI3.html</filename>
    <base>molpro::linalg::array::DistrArray::LocalBuffer</base>
    <member kind="function">
      <type></type>
      <name>LocalBufferMPI3</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1DistrArrayMPI3_1_1LocalBufferMPI3.html</anchorfile>
      <anchor>aface45b1617b7dd1333bb74458d2227c</anchor>
      <arglist>(DistrArrayMPI3 &amp;source)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::DistrArraySpan::LocalBufferSpan</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1DistrArraySpan_1_1LocalBufferSpan.html</filename>
    <base>molpro::linalg::array::DistrArray::LocalBuffer</base>
    <member kind="function">
      <type></type>
      <name>LocalBufferSpan</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1DistrArraySpan_1_1LocalBufferSpan.html</anchorfile>
      <anchor>a7d7d8bd6b2c978455fa25db5d3d9c86f</anchor>
      <arglist>(DistrArraySpan &amp;source)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LocalBufferSpan</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1DistrArraySpan_1_1LocalBufferSpan.html</anchorfile>
      <anchor>a172a15752368465c0b82d02be3f86522</anchor>
      <arglist>(const DistrArraySpan &amp;source)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::util::LockMPI3</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</filename>
    <class kind="struct">molpro::linalg::array::util::LockMPI3::Proxy</class>
    <member kind="function">
      <type></type>
      <name>LockMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>a58905b3592d0509c6a3a5d9a354ccff2</anchor>
      <arglist>(MPI_Comm comm)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~LockMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>a452b1101041cf943e1c88ed1b086de29</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LockMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>a1fccb7a6901b3c0f0e39b47721fb3266</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>LockMPI3</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>adeb507c610f38952db06e276d8b4d3d6</anchor>
      <arglist>(const LockMPI3 &amp;)=delete</arglist>
    </member>
    <member kind="function">
      <type>LockMPI3 &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>ae8924f20da0be2012d099e8f157460c4</anchor>
      <arglist>(const LockMPI3 &amp;)=delete</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>lock</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>a4837317958f1f5c9c804a2f21362b927</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>unlock</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>a174af52bb24cb7bcb3992e079476f9d8</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; Proxy &gt;</type>
      <name>scope</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>a58ddd065a72fff1550ecbc232f011fe4</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>MPI_Comm</type>
      <name>m_comm</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>a446f0f8b51d0559ecf1b996e9eaef9b9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>MPI_Win</type>
      <name>m_win</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>afcbde5f212701bfe2f935be5017b1efa</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_locked</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>a889a4fcc685ccdfd22e71ce0e1b430ec</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::weak_ptr&lt; Proxy &gt;</type>
      <name>m_proxy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3.html</anchorfile>
      <anchor>aaa8e7a1f50ca9eb6d2ce83237fcaff02</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::Logger</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</filename>
    <member kind="enumeration">
      <type></type>
      <name>Level</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>None</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a6e7a2261b6bfd1bc59dce5fe62c87d56</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Trace</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a4e0b021e4cf1dcd826c76bd7205072c6</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Debug</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0ad25d9df532ac44ba5d3d9cd40eaa4d06</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Info</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a3e71de103718f2811cea3f370e922dcd</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Warn</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0add79474a75277ef778e506b77c7fdd2f</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Error</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a819ab687a481cba7929cefb525913a10</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Fatal</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a61719e7779d976d0ec7051bc5f04ada6</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>None</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a6e7a2261b6bfd1bc59dce5fe62c87d56</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Trace</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a4e0b021e4cf1dcd826c76bd7205072c6</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Debug</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0ad25d9df532ac44ba5d3d9cd40eaa4d06</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Info</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a3e71de103718f2811cea3f370e922dcd</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Warn</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0add79474a75277ef778e506b77c7fdd2f</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Error</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a819ab687a481cba7929cefb525913a10</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>Fatal</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a8adb6b8c05cd17ed24547b35ed5cf4b0a61719e7779d976d0ec7051bc5f04ada6</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>msg</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a0a60cec13fb45bf89fc236ee3a541641</anchor>
      <arglist>(const std::string &amp;message, Level log_lvl)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>msg</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a548d9fd08f153e05e91053ddd97d11e9</anchor>
      <arglist>(const std::string &amp;message, ForwardIt begin, ForwardIt end, Level log_lvl, int precision=3)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static std::string</type>
      <name>scientific</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a2189cc66128accb5946097d3da89a956</anchor>
      <arglist>(double val)</arglist>
    </member>
    <member kind="variable">
      <type>Level</type>
      <name>max_trace_level</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>aae4f8d1afd45ee27292b55b11fef3b0e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Level</type>
      <name>max_warn_level</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>ab9d0869c7eaa4dbedadfb39518f95308</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>data_dump</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Logger.html</anchorfile>
      <anchor>a144df2151d6f7909f9f0c3c89488c6bf</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::mapped_or_value_type</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1mapped__or__value__type.html</filename>
    <templarg>class A</templarg>
    <templarg>bool</templarg>
    <member kind="typedef">
      <type>typename A::value_type</type>
      <name>type</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1mapped__or__value__type.html</anchorfile>
      <anchor>ac3af683434ca222a5e43ddc484bf1aa4</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::mapped_or_value_type&lt; A, true &gt;</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1mapped__or__value__type_3_01A_00_01true_01_4.html</filename>
    <templarg>class A</templarg>
    <member kind="typedef">
      <type>typename A::mapped_type</type>
      <name>type</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1mapped__or__value__type_3_01A_00_01true_01_4.html</anchorfile>
      <anchor>a590cfcd643692ada5ee8b39404c4585d</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::Matrix</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</filename>
    <templarg>typename T</templarg>
    <class kind="class">molpro::linalg::itsolv::subspace::Matrix::CSlice</class>
    <class kind="class">molpro::linalg::itsolv::subspace::Matrix::Slice</class>
    <member kind="typedef">
      <type>T</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a3bcee946103b44edb31292ff35bc62ba</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>size_t</type>
      <name>index_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>ae6763159839a6bde8a26bfe08f1ae8c1</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::pair&lt; size_t, size_t &gt;</type>
      <name>coord_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a41a73fef3548a5477eb9bb6c45838b05</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Matrix</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a72aff0a4f3f556d43931c3bf0d8a39e2</anchor>
      <arglist>(coord_type dims)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Matrix</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>aab5e505dc4bc4dadb5138e12e518c399</anchor>
      <arglist>(std::vector&lt; T &gt; &amp;&amp;data, coord_type dims)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Matrix</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a16e02b0929fc0e520f270b99ae525441</anchor>
      <arglist>(const std::vector&lt; T &gt; &amp;data, coord_type dims)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Matrix</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a8fbdbe1dd1780ddc018de1e40e568a66</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~Matrix</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a16dfc1d9362c476866d93365374e09ca</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Matrix</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a4d4abd30ac9cb7286a4d1ff55679c17d</anchor>
      <arglist>(const Matrix&lt; T &gt; &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Matrix</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>ab490ff99eea1d091909d899eedad316a</anchor>
      <arglist>(Matrix&lt; T &gt; &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; T &gt; &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a154eb1bae5b8eb16b22ee7e1086e2211</anchor>
      <arglist>(const Matrix&lt; T &gt; &amp;)=default</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; T &gt; &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a7ab9f9798446c8746bfd48fea28d49a8</anchor>
      <arglist>(Matrix&lt; T &gt; &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>T &amp;</type>
      <name>operator()</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a1c319c785894775225cd09d062da4dd8</anchor>
      <arglist>(index_type i, index_type j)</arglist>
    </member>
    <member kind="function">
      <type>T</type>
      <name>operator()</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a293991d7012f478ac86f7833d011b084</anchor>
      <arglist>(index_type i, index_type j) const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; T &gt; &amp;</type>
      <name>data</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a5b6284cf00334a7a6b73c374a38f8aeb</anchor>
      <arglist>() const &amp;</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; T &gt; &amp;&amp;</type>
      <name>data</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>af886fbff0733f847609a769512db6b28</anchor>
      <arglist>() &amp;&amp;</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>empty</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a999148333588f76643ddfa0501a78925</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>clear</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a60b9e3ac05ce61d9f1be1b8fe4d4c558</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>coord_type</type>
      <name>to_coord</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>ae4031b383d8c4578dd10216ded467dea</anchor>
      <arglist>(size_t ind) const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>ab0d94a27019942114d044437046ec7e7</anchor>
      <arglist>(T value)</arglist>
    </member>
    <member kind="function">
      <type>Slice</type>
      <name>slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a46b366c88de5a74104174a6198002329</anchor>
      <arglist>(coord_type upper_left, coord_type bottom_right)</arglist>
    </member>
    <member kind="function">
      <type>Slice</type>
      <name>slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a1a3e45c1cb20d09b43f49b535b3ecd55</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>CSlice</type>
      <name>slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a5473dfc4580dfea1cd875c5e2ec101ab</anchor>
      <arglist>(coord_type upper_left, coord_type bottom_right) const</arglist>
    </member>
    <member kind="function">
      <type>CSlice</type>
      <name>slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>aca310a12b0b680abf7e2e5238fa21176</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Slice</type>
      <name>row</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a2a1f8cb4dde583d42a7b4094b10e6733</anchor>
      <arglist>(size_t i)</arglist>
    </member>
    <member kind="function">
      <type>CSlice</type>
      <name>row</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a4a85720e021cab94c7f67f55535c9814</anchor>
      <arglist>(size_t i) const</arglist>
    </member>
    <member kind="function">
      <type>Slice</type>
      <name>col</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a3e3509d3f0f2c76550d61b91e1721681</anchor>
      <arglist>(size_t j)</arglist>
    </member>
    <member kind="function">
      <type>CSlice</type>
      <name>col</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a4e9e12601d628c87f99a2c108030b6a0</anchor>
      <arglist>(size_t j) const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>resize</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>ac6027c60134878de24d7937ebdd2b39a</anchor>
      <arglist>(const coord_type &amp;dims)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>remove_row</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a44ad56fc220c55d49864e61da80bdd0d</anchor>
      <arglist>(index_type row)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>remove_col</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a6378d894eaed58ace5c65e23fd587030</anchor>
      <arglist>(index_type col)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>remove_row_col</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a486219f03edd5ec5701001e855a8b6e7</anchor>
      <arglist>(index_type row, index_type col)</arglist>
    </member>
    <member kind="function">
      <type>index_type</type>
      <name>rows</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>adaf014c37112f0f0a60bf4f8bfe88c73</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>index_type</type>
      <name>cols</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>af86135c6710fa7edd2f521464e6386e9</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>coord_type</type>
      <name>dimensions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a66e50984a5564481e8e03577f57c4f38</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a7d13e0dbf1c95a0ef3fdffc0b2fbaaf0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>index_type</type>
      <name>m_rows</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a17a61a1afe2eea41c68b135597914f09</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>index_type</type>
      <name>m_cols</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>afe137ccc717ffd7a8de4ec0d43837538</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; T &gt;</type>
      <name>m_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix.html</anchorfile>
      <anchor>a47e15fb815a6fae994798da646ddc529</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>iterative_solver_matrix_problem::matrix_problem</name>
    <filename>structiterative__solver__matrix__problem_1_1matrix__problem.html</filename>
    <base>iterative_solver_problem::problem</base>
    <member kind="function">
      <type>procedure, pass</type>
      <name>attach</name>
      <anchorfile>structiterative__solver__matrix__problem_1_1matrix__problem.html</anchorfile>
      <anchor>ad3e7f40dd9b3fa1ad2158177f35159fe</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>rhs</name>
      <anchorfile>structiterative__solver__matrix__problem_1_1matrix__problem.html</anchorfile>
      <anchor>a9be92f8f1a735537f99d8ebf7ff5aa96</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>diagonals</name>
      <anchorfile>structiterative__solver__matrix__problem_1_1matrix__problem.html</anchorfile>
      <anchor>ab0d932f2cd05b9063f260222ad4753ed</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>action</name>
      <anchorfile>structiterative__solver__matrix__problem_1_1matrix__problem.html</anchorfile>
      <anchor>ac59b38994c3c8d34a0a2edb6d85edca2</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>pp_action_matrix</name>
      <anchorfile>structiterative__solver__matrix__problem_1_1matrix__problem.html</anchorfile>
      <anchor>adf41b7d9e3a624890c717a9707a9759f</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>p_action</name>
      <anchorfile>structiterative__solver__matrix__problem_1_1matrix__problem.html</anchorfile>
      <anchor>a36009c3afe206e116e1589314c66578c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double precision, dimension(:, :), pointer</type>
      <name>matrix</name>
      <anchorfile>structiterative__solver__matrix__problem_1_1matrix__problem.html</anchorfile>
      <anchor>a1f2b6aeb15895857cb5bf552d5acfd8a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double precision, dimension(:, :), pointer</type>
      <name>m_rhs</name>
      <anchorfile>structiterative__solver__matrix__problem_1_1matrix__problem.html</anchorfile>
      <anchor>a53f0fec7e5f7adf8290927ffec385d9e</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Morse_problem</name>
    <filename>classMorse__problem.html</filename>
    <base>molpro::linalg::itsolv::Problem</base>
    <member kind="function">
      <type></type>
      <name>Morse_problem</name>
      <anchorfile>classMorse__problem.html</anchorfile>
      <anchor>abfeeb8c897c7b1eed654f65e1bb7337e</anchor>
      <arglist>(Interpolate::point p0, Interpolate::point p1)</arglist>
    </member>
    <member kind="function">
      <type>value_t</type>
      <name>residual</name>
      <anchorfile>classMorse__problem.html</anchorfile>
      <anchor>abdd1b24df0694f03976f384ec57b4bbf</anchor>
      <arglist>(const R &amp;parameters, R &amp;residual) const override</arglist>
    </member>
  </compound>
  <compound kind="interface">
    <name>iterative_solver::mpi_init</name>
    <filename>interfaceiterative__solver_1_1mpi__init.html</filename>
  </compound>
  <compound kind="interface">
    <name>iterative_solver::mpi_rank_global</name>
    <filename>interfaceiterative__solver_1_1mpi__rank__global.html</filename>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::xspace::NewData</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace_1_1NewData.html</filename>
    <member kind="function">
      <type></type>
      <name>NewData</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace_1_1NewData.html</anchorfile>
      <anchor>ac3fb0fc2845c5ff018049ab157034433</anchor>
      <arglist>(size_t nQnew, size_t nX, size_t nRHS)</arglist>
    </member>
    <member kind="variable">
      <type>SubspaceData</type>
      <name>qq</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace_1_1NewData.html</anchorfile>
      <anchor>a66a2d505fb37b57b65c72bcee085677e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>SubspaceData</type>
      <name>qx</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace_1_1NewData.html</anchorfile>
      <anchor>a789820bcb8ea398dbdd20ac3857b7de5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>SubspaceData</type>
      <name>xq</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace_1_1NewData.html</anchorfile>
      <anchor>af2d362a7c1d45b75a57d790a06be77ed</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::NonLinearEquations</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquations.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>molpro::linalg::itsolv::IterativeSolver</base>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::NonLinearEquationsDIISOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsDIISOptions.html</filename>
    <base>molpro::linalg::itsolv::NonLinearEquationsOptions</base>
    <member kind="function">
      <type></type>
      <name>NonLinearEquationsDIISOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsDIISOptions.html</anchorfile>
      <anchor>a3dce7ee188b72a24685b8df631f52e99</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>NonLinearEquationsDIISOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsDIISOptions.html</anchorfile>
      <anchor>a417e97ff3a4b707992c7f4ebeed466cb</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>max_size_qspace</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsDIISOptions.html</anchorfile>
      <anchor>a30d5edbbc51efb65d4d79eb9fb75819c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>norm_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsDIISOptions.html</anchorfile>
      <anchor>aeb0cebaf5fd770cd5702c6d89f5bd3df</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>svd_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsDIISOptions.html</anchorfile>
      <anchor>a42b6faf3a6cd6f6a3e80310e9059c0d0</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::NonLinearEquationsOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsOptions.html</filename>
    <base>molpro::linalg::itsolv::Options</base>
    <member kind="function">
      <type></type>
      <name>NonLinearEquationsOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsOptions.html</anchorfile>
      <anchor>ad3052dd32b3080134143f669caad0cac</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>NonLinearEquationsOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1NonLinearEquationsOptions.html</anchorfile>
      <anchor>a859bf7c43557377b6c18672a18a6b13b</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::OperationRegister</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1OperationRegister.html</filename>
    <templarg>typename... Args</templarg>
    <member kind="typedef">
      <type>std::tuple&lt; Args... &gt;</type>
      <name>OP</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1OperationRegister.html</anchorfile>
      <anchor>adec453c471eab2cf48d7c8c43a9e31c2</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>push</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1OperationRegister.html</anchorfile>
      <anchor>a85d5441e4a6d527c6672878ab2dad9f1</anchor>
      <arglist>(const Args &amp;...args, ArgEqual equal)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>push</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1OperationRegister.html</anchorfile>
      <anchor>adf738d84096126b2c6442399eafebd8a</anchor>
      <arglist>(const Args &amp;...args)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>empty</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1OperationRegister.html</anchorfile>
      <anchor>aa0c23529a797aa3a5f2c0b2926cba03f</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>clear</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1OperationRegister.html</anchorfile>
      <anchor>a91d2643100e88fafe2a373d4ba66f9b9</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable">
      <type>std::list&lt; std::tuple&lt; Args... &gt; &gt;</type>
      <name>m_register</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1OperationRegister.html</anchorfile>
      <anchor>a10215e9ae57e6cede38da1ce2e899022</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::Optimize</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1Optimize.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>molpro::linalg::itsolv::IterativeSolver</base>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::OptimizeBFGS</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>IterativeSolverTemplate&lt; Optimize, R, R, std::map&lt; size_t, typename R::value_type &gt; &gt;</base>
    <member kind="typedef">
      <type>IterativeSolverTemplate&lt; Optimize, R, Q, P &gt;</type>
      <name>SolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>ab150d2cc70608566f7ae8c779494b493</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>subspace::SubspaceSolverOptBFGS&lt; R, Q, P &gt;</type>
      <name>SubspaceSolver</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a9c2255686f67f7ab9f938dc2e21f95c2</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>OptimizeBFGS</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>ad95a325bacae4e84d6cbdf33c03231ca</anchor>
      <arglist>(const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers, const std::shared_ptr&lt; Logger &gt; &amp;logger_=std::make_shared&lt; Logger &gt;())</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>nonlinear</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a33d00ccc2b14f572705f881f99e44b41</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>add_vector</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a06006188007b3f5240f089a813019b39</anchor>
      <arglist>(R &amp;parameters, R &amp;residual, value_type value) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>BFGS_update_1</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a27e9a19a8c2a15222c173d48bfdbcee7</anchor>
      <arglist>(R &amp;residual, std::shared_ptr&lt; const subspace::IXSpace&lt; R, Q, P &gt; &gt; xspace, const Matrix&lt; double &gt; &amp;H)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>BFGS_update_2</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a6ec030db3fc6b6055169c4f7f21dfa41</anchor>
      <arglist>(R &amp;z, std::shared_ptr&lt; const subspace::IXSpace&lt; R, Q, P &gt; &gt; xspace, const Matrix&lt; double &gt; &amp;H)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>aadfaf83aa684dac364076cd1a593c77f</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a5e182b874043a52cc77b1acc0ef02e0e</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a129cec6f2f2ad88a8ee2df0b18659991</anchor>
      <arglist>(R &amp;parameters, R &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_value_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a811aacdf75f1e83e8b1ab9c95cd01a66</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a99fca8a245026032ba0f92784883df1e</anchor>
      <arglist>(int n)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>get_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>ab170d17ad464245443decb12d558b1d9</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a6e35691eacd5524090e51113a4965520</anchor>
      <arglist>(const Options &amp;options) override</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; Options &gt;</type>
      <name>get_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a3310a0da2e529678642d9ff058fc7eca</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a2d0c85e8b671a68c34fb1665aeb97aef</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a2586061786c83f40ee932189f6a59353</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>ab4748964a82d882be6cfebd7c158511e</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="variable">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a44fbe5b4ca9b11639e77170bf7c4ba9a</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>construct_residual</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>ab20fe2bb9557a3d2c1f41ce3f87ebe87</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const CVecRef&lt; R &gt; &amp;params, const VecRef&lt; R &gt; &amp;actions) override</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; double &gt;</type>
      <name>m_BFGS_update_alpha</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a706ec8e6979add2a97617a91fd85e6c1</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_linesearch</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a415a0fa132884a5a0e1b7d06f51a635f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_last_iteration_linesearching</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>aba57f96b3db491cbc5a9b5470ad1067c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_max_size_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a4d6b245df41c99eaa36f057d8e8a47a7</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_strong_Wolfe</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a0df9462c03d7bd7aa35819b3b60b6b21</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_Wolfe_1</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a20ec1c340efd4c780f38320896acf8cb</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_Wolfe_2</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a19014fc6fc3451bfa09960ae2b905ffd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_linesearch_tolerance</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a2f6898258c8fcd943a4779b8eb8e740c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_linesearch_grow_factor</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>af4fc19800b03d96ce7185daafe8c84a1</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_quasinewton_maximum_step</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGS.html</anchorfile>
      <anchor>a4ea11459f591fea9ce30b9ac3d3dab66</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::OptimizeBFGSOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</filename>
    <base>molpro::linalg::itsolv::OptimizeOptions</base>
    <member kind="function">
      <type></type>
      <name>OptimizeBFGSOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>ac4f63d5c28880e35d9246ae34e51688e</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>OptimizeBFGSOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>a00deb0bf30e5055ba13d4a7bc2fff26a</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>max_size_qspace</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>aa6d77eb00715031a13b4fa721819e6c5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>norm_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>a3a8c26639074dfe38a89cfeac3c7a15f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>svd_thresh</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>adf37f6cd71efda9948e29f1731e4701e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; bool &gt;</type>
      <name>strong_Wolfe</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>ad51a99040ec9f0fecd6bf2c55947535b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>Wolfe_1</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>a5c907211361ce4d9f8ae13bd08973a8b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>Wolfe_2</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>a5c3f9e9840f1919c8efe9aa1c9ea38c9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>linesearch_tolerance</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>ac66b95b29d700bbf65b35385be58cc13</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>linesearch_grow_factor</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>af0caac94a3d9e2dbc7080de6a2d69053</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>quasinewton_maximum_step</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeBFGSOptions.html</anchorfile>
      <anchor>a10077924e329f444e1e18b30441734ef</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::OptimizeOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeOptions.html</filename>
    <base>molpro::linalg::itsolv::Options</base>
    <member kind="function">
      <type></type>
      <name>OptimizeOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeOptions.html</anchorfile>
      <anchor>af15b21dd77edf275a0ce93328ec2ff3a</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>OptimizeOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeOptions.html</anchorfile>
      <anchor>af818b1286046e4876d81c6c714e88a5d</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::OptimizeSD</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>IterativeSolverTemplate&lt; Optimize, R, R, std::map&lt; size_t, typename R::value_type &gt; &gt;</base>
    <member kind="typedef">
      <type>IterativeSolverTemplate&lt; Optimize, R, Q, P &gt;</type>
      <name>SolverTemplate</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a0f558dd0ad9d7da386a0189ec22bb3eb</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>subspace::SubspaceSolverOptSD&lt; R, Q, P &gt;</type>
      <name>SubspaceSolver</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a2fee82c5ea4b4dc66f9e0173ba1c7339</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>OptimizeSD</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>aa083792a8df576412f41879fbad9a164</anchor>
      <arglist>(const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers, const std::shared_ptr&lt; Logger &gt; &amp;logger_=std::make_shared&lt; Logger &gt;())</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>nonlinear</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a5471155b5ff63e8206823da6afe27ebe</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a59977558b57c095960ca116bb7b1f3c7</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a713f270c65bfedc5aa91fe096aa25d79</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;action) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_value_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a59cfae007796b412274f8a6b1761ce28</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>ad55d168c0b41759afc65e772fb28f88d</anchor>
      <arglist>(const Options &amp;options) override</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; Options &gt;</type>
      <name>get_options</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a8fb8626f4484de8f65d884ac0ce7bd12</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a3a45ba4bb4649f7d4c5036b288e8060b</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>add_vector</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a8976b993c5c2a8fe53a9b40898ea4626</anchor>
      <arglist>(R &amp;parameters, R &amp;residual, value_type value) override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>end_iteration</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a200c4935c6e218cf3e7bf80bb70e4646</anchor>
      <arglist>(R &amp;parameters, R &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a2586061786c83f40ee932189f6a59353</anchor>
      <arglist>(std::ostream &amp;cout, bool endl=true) const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>report</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>ab4748964a82d882be6cfebd7c158511e</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="variable">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a62a4f2dc5d5fc2d3cc93d8d562612c3a</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>construct_residual</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1OptimizeSD.html</anchorfile>
      <anchor>a2a0132a2b2e6a29edf87f68f514e426d</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const CVecRef&lt; R &gt; &amp;params, const VecRef&lt; R &gt; &amp;actions) override</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::OptimizeSDOptions</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeSDOptions.html</filename>
    <base>molpro::linalg::itsolv::OptimizeOptions</base>
    <member kind="function">
      <type></type>
      <name>OptimizeSDOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeSDOptions.html</anchorfile>
      <anchor>a13e6320fbbf9725456145b536a84f1a0</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>OptimizeSDOptions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1OptimizeSDOptions.html</anchorfile>
      <anchor>a03d16dfaeddd18aa508fb47ff327349c</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::Options</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</filename>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~Options</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>af5cfb94d09d44e75af4defd698b1645a</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Options</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>a37831f7f1c9e8ca25231cf360b9dadc1</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Options</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>adf4361108ecfcb3699596d68b348af59</anchor>
      <arglist>(const options_map &amp;opt)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>a2b3e52ac368db1f195ee7b1f437967d5</anchor>
      <arglist>(const Options &amp;source)</arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>convergence_threshold</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>a8301b90cd1b614ee591695e194d40bfd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>n_roots</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>afe2a175b7e8c9c864a7ed477ae3d3dcc</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; Verbosity &gt;</type>
      <name>verbosity</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>af18f1e9188ab91061d7d5efef5b4cf24</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; int &gt;</type>
      <name>max_iter</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>ad2c23ab9becc59f81d30b10080deb732</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>max_p</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>abdcdb4d71ce0a3b9e2ab82e84420059e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::optional&lt; double &gt;</type>
      <name>p_threshold</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Options.html</anchorfile>
      <anchor>a63ef706a59c01216999aa65ddf6a0f80</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::util::detail::Overlap</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1Overlap.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class Z</templarg>
    <templarg>class W</templarg>
    <templarg>bool</templarg>
    <templarg>bool</templarg>
    <templarg>bool</templarg>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::util::detail::Overlap&lt; R, Q, Z, W, true, false, false &gt;</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1Overlap_3_01R_00_01Q_00_01Z_00baa0799f1276d81be52429d09f0018d4.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class Z</templarg>
    <templarg>class W</templarg>
    <member kind="function" static="yes">
      <type>static Matrix&lt; double &gt;</type>
      <name>_</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1Overlap_3_01R_00_01Q_00_01Z_00baa0799f1276d81be52429d09f0018d4.html</anchorfile>
      <anchor>a3153134216283283858a9114d49c0ea4</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;left, const CVecRef&lt; Q &gt; &amp;right, array::ArrayHandler&lt; Z, W &gt; &amp;handler)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::util::detail::Overlap&lt; R, Q, Z, W, true, true, true &gt;</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1Overlap_3_01R_00_01Q_00_01Z_009b17c8d430e8f336e99dfcdf84344305.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class Z</templarg>
    <templarg>class W</templarg>
    <member kind="function" static="yes">
      <type>static Matrix&lt; double &gt;</type>
      <name>_</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail_1_1Overlap_3_01R_00_01Q_00_01Z_009b17c8d430e8f336e99dfcdf84344305.html</anchorfile>
      <anchor>ac61631682766de14efe854a9c090a1d9</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;left, const CVecRef&lt; Q &gt; &amp;right, array::ArrayHandler&lt; Z, W &gt; &amp;handler)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::Interpolate::point</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1Interpolate_1_1point.html</filename>
    <member kind="variable">
      <type>double</type>
      <name>x</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Interpolate_1_1point.html</anchorfile>
      <anchor>a095f7815af290da3e6fe2e6f5d66b3a4</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>f</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Interpolate_1_1point.html</anchorfile>
      <anchor>a2371f9b68ceea55396112011591b53e9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>f1</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Interpolate_1_1point.html</anchorfile>
      <anchor>a348638e433f5410f2f9b92a7237888c2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>f2</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Interpolate_1_1point.html</anchorfile>
      <anchor>a236f1245ef8db0d33fa9221a34e4a279</anchor>
      <arglist></arglist>
    </member>
    <member kind="friend">
      <type>friend std::ostream &amp;</type>
      <name>operator&lt;&lt;</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Interpolate_1_1point.html</anchorfile>
      <anchor>ad3e691710c53fd3bc3cdbed8850e3d51</anchor>
      <arglist>(std::ostream &amp;os, const point &amp;p)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>iterative_solver_problem::problem</name>
    <filename>structiterative__solver__problem_1_1problem.html</filename>
    <member kind="function">
      <type>procedure, pass</type>
      <name>diagonals</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>ad7241abf6805551d1dc0b2676aecd9e9</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>precondition</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>ad14be3f894acae5a09e59d46b9eff8ef</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>residual</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>adf9c229878bfe6bac1c0e0967a94428c</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>action</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>a061ed74cbe7993c76ccee00b75218a93</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>rhs</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>a5134b99305b8e07054d0a3e7eb0c246d</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>report</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>ab226e906c11bbee23169e73ca6f12280</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>p_action</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>a284e747ad56ea8980eaf65123547db8a</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>pp_action_matrix</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>a92f51ae6a758597adc86539eb92e10f9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>type(pspace)</type>
      <name>p_space</name>
      <anchorfile>structiterative__solver__problem_1_1problem.html</anchorfile>
      <anchor>a0ee091c82a63ba2ffe604fe5713c79eb</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::Problem</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</filename>
    <templarg>typename R</templarg>
    <templarg>typename P</templarg>
    <member kind="typedef">
      <type>R</type>
      <name>container_t</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>a10aafe6ea77b7b593ae4803b65adbe63</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename R::value_type</type>
      <name>value_t</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>a4f65ebe42fc79462292b72586c342e63</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Problem</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>a4bbcd88d13af305c361523c22e78dbed</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~Problem</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>a5bf522d4b55091290e5d5a28bb7b4e9c</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual value_t</type>
      <name>residual</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>ac8af88717d898dce1e6ae0884142ea6d</anchor>
      <arglist>(const R &amp;parameters, R &amp;residual) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>action</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>a0d3d1df78eda376e661d9677449008ca</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;action) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual bool</type>
      <name>diagonals</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>a6887ee4a7b23709c1164ebcf5221ce8e</anchor>
      <arglist>(container_t &amp;d) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>precondition</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>a48be93529023c863f0631a2bcbafb64c</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;residual, const std::vector&lt; value_t &gt; &amp;shift) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>precondition</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>af0feba50834c98b633107e407900535b</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;residual, const std::vector&lt; value_t &gt; &amp;shift, const R &amp;diagonals) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual bool</type>
      <name>RHS</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>ae783f40d90e529325975408e94c3291c</anchor>
      <arglist>(R &amp;RHS, unsigned int instance) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::vector&lt; double &gt;</type>
      <name>pp_action_matrix</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>a3cc421a58afcaceb42a0631828221cdd</anchor>
      <arglist>(const std::vector&lt; P &gt; &amp;pparams) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>p_action</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>aa2d057b7e6b50596cf0f3206a3d4d2a8</anchor>
      <arglist>(const std::vector&lt; std::vector&lt; value_t &gt; &gt; &amp;p_coefficients, const CVecRef&lt; P &gt; &amp;pparams, const VecRef&lt; container_t &gt; &amp;actions) const</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual bool</type>
      <name>test_parameters</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1Problem.html</anchorfile>
      <anchor>ae2396d45227f1ffaa7f71da4a5087f1f</anchor>
      <arglist>(unsigned int instance, R &amp;parameters) const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::util::DistrFlags::Proxy</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</filename>
    <member kind="function">
      <type></type>
      <name>Proxy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a347c0feba988c1b0067b928d775f6819</anchor>
      <arglist>(MPI_Comm comm, MPI_Win win, int rank, std::shared_ptr&lt; int &gt; counter)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~Proxy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a56d5d9040bc0a3b0e47810f694187c67</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>get</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a6de16fdcae3045ffb5354bdc943f2877</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>replace</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a6b792fba2e50251c0ea098be902534f4</anchor>
      <arglist>(int val)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>rank</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a576577ffaac05fb9c9015ef0eea955d1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>MPI_Comm</type>
      <name>m_comm</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a0fe0863b1d920893eafe00fb23cdbc99</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>MPI_Win</type>
      <name>m_win</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a3ff881e650e31379ce435fbda709650b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>int</type>
      <name>m_rank</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a6aa0ebd7cd3f2aa705eab462787d5c12</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; int &gt;</type>
      <name>m_counter</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1DistrFlags_1_1Proxy.html</anchorfile>
      <anchor>a532f507776301d9f4a0039082efcb326</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::LockMPI3::Proxy</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3_1_1Proxy.html</filename>
    <member kind="function">
      <type></type>
      <name>Proxy</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3_1_1Proxy.html</anchorfile>
      <anchor>a99dd29b7a12cc785474c0936dd4335da</anchor>
      <arglist>(LockMPI3 &amp;)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Proxy</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3_1_1Proxy.html</anchorfile>
      <anchor>af58e970c26bd9dac6a5619d9c6efb190</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~Proxy</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3_1_1Proxy.html</anchorfile>
      <anchor>a9fd46ffa600db9a7cfbe0278a11b1639</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable">
      <type>LockMPI3 &amp;</type>
      <name>m_lock</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3_1_1Proxy.html</anchorfile>
      <anchor>a33f11d90da05eb6a7b5fec81d141b141</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>m_deleted</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1LockMPI3_1_1Proxy.html</anchorfile>
      <anchor>a47814fe7fa063f47e1969eea29016bf1</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::ArrayHandlers::Builder::Proxy</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder_1_1Proxy.html</filename>
    <templarg>class T</templarg>
    <templarg>class S</templarg>
    <member kind="function">
      <type></type>
      <name>Proxy</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder_1_1Proxy.html</anchorfile>
      <anchor>af7e20eecbf6af61f0e8380eafd5451de</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Proxy</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder_1_1Proxy.html</anchorfile>
      <anchor>a34f81a70e437c714c5df45be102ff85b</anchor>
      <arglist>(Builder *b)</arglist>
    </member>
    <member kind="function">
      <type>Builder &amp;</type>
      <name>operator()</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder_1_1Proxy.html</anchorfile>
      <anchor>a4201b5d57302c1d37eb32fcb4f0381db</anchor>
      <arglist>(const std::shared_ptr&lt; array::ArrayHandler&lt; T, S &gt; &gt; &amp;h)</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; T, S &gt; &gt;</type>
      <name>handler</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder_1_1Proxy.html</anchorfile>
      <anchor>ad3c2a9e4ce0b39b68faf7c96c11ec1b1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Builder *</type>
      <name>builder</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder_1_1Proxy.html</anchorfile>
      <anchor>a73c17387df89526fc0bcbb52efc09a03</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; array::ArrayHandler&lt; T, S &gt; &gt;</type>
      <name>m_handler</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1ArrayHandlers_1_1Builder_1_1Proxy.html</anchorfile>
      <anchor>a7604116fa565960dc5f9866fc5907b61</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::ArrayHandler::ProxyHandle</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</filename>
    <member kind="function">
      <type></type>
      <name>ProxyHandle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>a1fe711bb00bfc533ec16b1f663d4b124</anchor>
      <arglist>(std::shared_ptr&lt; LazyHandle &gt; handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>a5daab75b9b2b8173e8787a734f5b521b</anchor>
      <arglist>(Args &amp;&amp;...args)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>dot</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>aa9281d9e60eeb5151eebf30fc3065b14</anchor>
      <arglist>(Args &amp;&amp;...args)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>eval</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>a3fbc8e8cf5973a81ee0dfff9f32aff74</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>invalidate</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>a7666eaa8bcf4a9a1997e1f86dc76a24e</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>invalid</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>a94ae88f7320b91c880474a5ecae16d94</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>off</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>ad0091fda4de382bbbe70a45e97a9e4a5</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>on</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>ac075222a6329f35419e439a2ccadf36c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>is_off</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>a00385a3163d3f2a6e6fd12bc95c90a04</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; LazyHandle &gt;</type>
      <name>m_lazy_handle</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>afbcf9562ef99b911a7a41617a1613b99</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_off</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1ArrayHandler_1_1ProxyHandle.html</anchorfile>
      <anchor>a9afd8b7fefd3a00507666f8197c0f934</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>iterative_solver_pspace::pspace</name>
    <filename>structiterative__solver__pspace_1_1pspace.html</filename>
    <member kind="function">
      <type>procedure, pass</type>
      <name>add_complex</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>a70bb8096bb364df84977c4e6291e99e6</anchor>
      <arglist>=&gt; pspace_add_complex</arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>add_simple</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>a0d239e70d6b9d6cbc8618859b4f91952</anchor>
      <arglist>=&gt; pspace_add_simple</arglist>
    </member>
    <member kind="function">
      <type>procedure, pass</type>
      <name>ensure</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>a49827a40cea8691669ce6918e112729e</anchor>
      <arglist>=&gt; pspace_ensure</arglist>
    </member>
    <member kind="function">
      <type>final</type>
      <name>pspace_final</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>a0fb398b66fd222ec8ce0e31879d431d9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>integer, dimension(:), allocatable</type>
      <name>indices</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>a68c4f66b40654f8a3b47b632a5ab07aa</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>integer, dimension(:), allocatable</type>
      <name>offsets</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>ac73ce551415e7871f146b2ae06aea58e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double precision, dimension(:), allocatable</type>
      <name>coefficients</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>ab401c8d6b3adfe8a8a94d1e899ddfa51</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>integer</type>
      <name>size</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>a424431302b02c7b600beffee763a70be</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>logical</type>
      <name>simple</name>
      <anchorfile>structiterative__solver__pspace_1_1pspace.html</anchorfile>
      <anchor>a31ded964a3790fad6862674ec5822a14</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::PSpace</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</filename>
    <templarg>class Rt</templarg>
    <templarg>class Pt</templarg>
    <member kind="typedef">
      <type>Rt</type>
      <name>R</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</anchorfile>
      <anchor>a5a84e07e35aedf62ca9e3a5e96c7402f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>Pt</type>
      <name>P</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</anchorfile>
      <anchor>aa34744b4a0522bd2d428fd01bd735057</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>update</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</anchorfile>
      <anchor>a2bcdd4d81a372ef031794d0188a57496</anchor>
      <arglist>(const CVecRef&lt; P &gt; &amp;params, array::ArrayHandler&lt; P, P &gt; &amp;handler)</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; P &gt;</type>
      <name>params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</anchorfile>
      <anchor>a47a2aac4420663059179863792c096a0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; P &gt;</type>
      <name>cparams</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</anchorfile>
      <anchor>aad9bf852d32d7d6717977d0504204f86</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; P &gt;</type>
      <name>params</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</anchorfile>
      <anchor>a09ce15c76a23e50a56a8454e81e0bc8d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</anchorfile>
      <anchor>a5776c692fb930d06a6375fd2c500e4ba</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>erase</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1PSpace.html</anchorfile>
      <anchor>aa79849c9fd557a4d1cda0f8972f9c886</anchor>
      <arglist>(size_t i)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::qspace::QParam</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1qspace_1_1QParam.html</filename>
    <templarg>class Q</templarg>
    <member kind="variable">
      <type>std::unique_ptr&lt; Q &gt;</type>
      <name>param</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1qspace_1_1QParam.html</anchorfile>
      <anchor>a160e66d4cc94927f248b425abba0650d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::unique_ptr&lt; Q &gt;</type>
      <name>action</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1qspace_1_1QParam.html</anchorfile>
      <anchor>a440925a373067a1b7d6ead39904c4655</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>id</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1qspace_1_1QParam.html</anchorfile>
      <anchor>a722b1cc4cd16631c50eaff379c885dc8</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::subspace::QSpace</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</filename>
    <templarg>class Rt</templarg>
    <templarg>class Qt</templarg>
    <templarg>class Pt</templarg>
    <member kind="typedef">
      <type>Rt</type>
      <name>R</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>aa807634512d7dc84132cc3f827360cb0</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>Qt</type>
      <name>Q</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>ab428b56edffdf7ea835a6a13da470e84</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>Pt</type>
      <name>P</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a543770b2a94132265a586d0833aa0f04</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; R, R &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a4cf7408d125c98bf680433b5cf258c78</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename array::ArrayHandler&lt; R, R &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a052256068f73180ecd2a422345f26c73</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>QSpace</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>acb898c8d68927cb03e9c00f027940e14</anchor>
      <arglist>(std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; handlers, std::shared_ptr&lt; Logger &gt; logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>update</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a2e4f93b4f458e9e7ebea7cfa257e142f</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;params, const CVecRef&lt; R &gt; &amp;actions, const SubspaceData &amp;qq, const SubspaceData &amp;qx, const SubspaceData &amp;xq, const Dimensions &amp;dims, SubspaceData &amp;old_data)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>clear</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a68224e6e8fd5575da228740d37b432cb</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>erase</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a8ef0f75cf99589420689978b0c48b839</anchor>
      <arglist>(size_t i)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a643496b7f661c75d58fe7f708ad54d62</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; Q &gt;</type>
      <name>params</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a35528d743c7bd9cf3e54226acae82eb3</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>params</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>ae3f19d0ffcc348a9c3ec5209d9c82afb</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>cparams</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a724ca901c88aaefd54e121835341c24d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; Q &gt;</type>
      <name>actions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>abc3eef3f8208d4cb8ed8f01b63234571</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>actions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>ae749c6bd56eb739a1117bce4cb335491</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>cactions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a50ac834a2564d9e52258235b0d556e67</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt;</type>
      <name>m_handlers</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a2c537128d066ee283b8d047b21706a52</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>m_logger</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>ab31bb8c1155454ec98ec10b3dd71af28</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>size_t</type>
      <name>m_unique_id</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a8ca43df281627ede9ae7f4cb49c019a2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::list&lt; qspace::QParam&lt; Q &gt; &gt;</type>
      <name>m_params</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1QSpace.html</anchorfile>
      <anchor>a98a32594c252cf4d39abbfdbf569462e</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::RefEqual</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1RefEqual.html</filename>
    <templarg>typename T</templarg>
    <member kind="function">
      <type>bool</type>
      <name>operator()</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1RefEqual.html</anchorfile>
      <anchor>a1d71d8c5238f6cddb1a15db36ecbb882</anchor>
      <arglist>(const std::reference_wrapper&lt; T &gt; &amp;l, const std::reference_wrapper&lt; T &gt; &amp;r)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::util::ScopeLock</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1util_1_1ScopeLock.html</filename>
    <member kind="function">
      <type></type>
      <name>ScopeLock</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1ScopeLock.html</anchorfile>
      <anchor>a88960f9abaefc06403a2b80f07d5827b</anchor>
      <arglist>(MPI_Comm comm)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>LockMPI3</type>
      <name>lock</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1ScopeLock.html</anchorfile>
      <anchor>a252826f8aa4d2d17d6c08fba66eeaca9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>decltype(std::declval&lt; LockMPI3 &gt;().scope())</type>
      <name>l</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1ScopeLock.html</anchorfile>
      <anchor>afc9adb1f67144de0e945c6f741c2b513</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::Matrix::Slice</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</filename>
    <member kind="function">
      <type></type>
      <name>Slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a414b7de61feb1e46a28741e4260bbe82</anchor>
      <arglist>(Matrix&lt; T &gt; &amp;matrix, coord_type upper_left, coord_type bottom_right)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a9005e83f19e9f53069a59d5bf78a96a7</anchor>
      <arglist>()=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~Slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>af3ec7003179e40a9341a6713303131a1</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a4da12705d53db72961ac7a0ebb214a36</anchor>
      <arglist>(const Slice &amp;)=delete</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Slice</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a155b5568d037955ce1be7c7bd3bf4d8d</anchor>
      <arglist>(Slice &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>Slice &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a6015765fef88e74c33946a5c4d4fa8cb</anchor>
      <arglist>(Slice &amp;&amp;) noexcept=default</arglist>
    </member>
    <member kind="function">
      <type>Slice &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>aba7f457905383a6cb7013cb17003a6e3</anchor>
      <arglist>(const Slice &amp;right)</arglist>
    </member>
    <member kind="function">
      <type>T &amp;</type>
      <name>operator()</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a662361ab16ddca5fbd1b1f3738d7229c</anchor>
      <arglist>(size_t i, size_t j)</arglist>
    </member>
    <member kind="function">
      <type>T</type>
      <name>operator()</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a61244ceb1487474bd1fb855102b522c4</anchor>
      <arglist>(size_t i, size_t j) const</arglist>
    </member>
    <member kind="function">
      <type>Slice &amp;</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>aee4cf0eb06a19ad753851e497d7d12bb</anchor>
      <arglist>(T a, const Slice &amp;x)</arglist>
    </member>
    <member kind="function">
      <type>Slice &amp;</type>
      <name>axpy</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>ae0e9eedaa82992892a906dfa092b5b59</anchor>
      <arglist>(T a, const CSlice &amp;x)</arglist>
    </member>
    <member kind="function">
      <type>Slice &amp;</type>
      <name>scal</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>ae0f74c8995df66be6d3811c47d85b622</anchor>
      <arglist>(T a)</arglist>
    </member>
    <member kind="function">
      <type>Slice &amp;</type>
      <name>fill</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a40afc54ce44d353e318a0caa1b01d101</anchor>
      <arglist>(T a)</arglist>
    </member>
    <member kind="function">
      <type>Slice &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a1248d20a80cc68b291edf9917a0d3924</anchor>
      <arglist>(const CSlice &amp;right)</arglist>
    </member>
    <member kind="function">
      <type>Slice &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a841d75dd1d6db5b609f50078e2180a29</anchor>
      <arglist>(const Matrix&lt; T &gt; &amp;right)</arglist>
    </member>
    <member kind="function">
      <type>coord_type</type>
      <name>dimensions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a1d8e8768d59bfe7d9be494bf62a9c732</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>rows</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a4f5b5940135cca6829a2b9ac00fbcc51</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>cols</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>aa80ee67d6c2b1477f3177969268df7f1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Matrix&lt; T &gt; &amp;</type>
      <name>mat</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a3c43277facb126e13505bd932435c523</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>coord_type</type>
      <name>upl</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a0769807933c25ef3dffa52ccfd16b832</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>coord_type</type>
      <name>btr</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1Matrix_1_1Slice.html</anchorfile>
      <anchor>a668c9d703d82dfd8298fe288b9540d94</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::SolverFactory</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1SolverFactory.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~SolverFactory</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1SolverFactory.html</anchorfile>
      <anchor>a9df8895d7401f31ea43f324538bddcd3</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::unique_ptr&lt; IterativeSolver&lt; R, Q, P &gt; &gt;</type>
      <name>create</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1SolverFactory.html</anchorfile>
      <anchor>a642c9f3fe829bd137e0eb65ea32801cc</anchor>
      <arglist>(const Options &amp;options, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::unique_ptr&lt; LinearEigensystem&lt; R, Q, P &gt; &gt;</type>
      <name>create</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1SolverFactory.html</anchorfile>
      <anchor>abfbcd5624f05c8f86e335cc99b9d50ee</anchor>
      <arglist>(const LinearEigensystemOptions &amp;options, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::unique_ptr&lt; LinearEquations&lt; R, Q, P &gt; &gt;</type>
      <name>create</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1SolverFactory.html</anchorfile>
      <anchor>a54fb97c0acd0474c5d06133894ce5291</anchor>
      <arglist>(const LinearEquationsOptions &amp;options, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::unique_ptr&lt; NonLinearEquations&lt; R, Q, P &gt; &gt;</type>
      <name>create</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1SolverFactory.html</anchorfile>
      <anchor>a2a1b764692aac77d6269cb291a528eab</anchor>
      <arglist>(const NonLinearEquationsOptions &amp;options=NonLinearEquationsOptions{}, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::unique_ptr&lt; Optimize&lt; R, Q, P &gt; &gt;</type>
      <name>create</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1SolverFactory.html</anchorfile>
      <anchor>a50e2484ee9780cd46d7c6d5ee24e642d</anchor>
      <arglist>(const OptimizeOptions &amp;options=OptimizeOptions{}, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::unique_ptr&lt; IterativeSolver&lt; R, Q, P &gt; &gt;</type>
      <name>create</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1SolverFactory.html</anchorfile>
      <anchor>a564027dd4f9d2e325fb8b785a5347aef</anchor>
      <arglist>(const std::string &amp;method, const options_map &amp;options, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::span::Span</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</filename>
    <templarg>typename T</templarg>
    <member kind="typedef">
      <type>T</type>
      <name>element_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a7bf89af3ae743e5310748f1dceea5a21</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::remove_cv_t&lt; T &gt;</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a03602cebfacc24ff12bb706a2dbe1436</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>T &amp;</type>
      <name>reference</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a3d32f877f231256a1ca941159569e9a4</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>T &amp;</type>
      <name>const_reference</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a823c7414acdf43ed7bcfd15867f3233c</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>size_t</type>
      <name>size_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a8b3a38440fadc5f5f1cc8e6b790b51df</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::ptrdiff_t</type>
      <name>difference_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>afd7dc63bf284ae849ea34adc5ba6b822</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>T *</type>
      <name>iterator</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>af3f39b373eef1afacdcc0e9b984102ca</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>T const  *</type>
      <name>const_iterator</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a2aadc87f78023f963bd48b6898517cf1</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Span</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a21f614cddfbfcf554e33ee2e288bd4bd</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~Span</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a349c7ea95f10e82b45e5133ec194e094</anchor>
      <arglist>()=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Span</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a1f47386e9f38d285d6116650ab78b5a5</anchor>
      <arglist>(T *data, size_type size)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Span</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>ad6dea4305315afc49f427824e90d1ed4</anchor>
      <arglist>(const Span &amp;source)=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Span</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a2f868dfe54168f3261c3f71b36915149</anchor>
      <arglist>(Span&lt; T &gt; &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type>Span &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>aee9e7c2a85bcc24500574de307ca9d4c</anchor>
      <arglist>(const Span &amp;source)=default</arglist>
    </member>
    <member kind="function">
      <type>Span &amp;</type>
      <name>operator=</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a85ddb4ea7a0c899fa40e4dd19e56a162</anchor>
      <arglist>(Span &amp;&amp;source) noexcept</arglist>
    </member>
    <member kind="function">
      <type>reference</type>
      <name>operator[]</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a61d6f5a40fafa2d66852986b563f0184</anchor>
      <arglist>(size_type i)</arglist>
    </member>
    <member kind="function">
      <type>const_reference</type>
      <name>operator[]</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a5f437126d38a51c39b8cc18aea81369f</anchor>
      <arglist>(size_type i) const</arglist>
    </member>
    <member kind="function">
      <type>iterator</type>
      <name>data</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>af3d8343f18111df965fa0fd08d60a237</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>data</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>abbea03323566f407633b041dac09842f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>iterator</type>
      <name>begin</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>ac02b118c484b8c079c0c19d61ec209af</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>begin</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>aa9594956602678fdad99d87de70bbff1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>cbegin</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a4ce517815c2a359455d496ad892b24e1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>iterator</type>
      <name>end</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>ad010e3e7b4772da2fff82f074866f89c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>end</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a6fd0a5fb7d0920d9444c0bd2c8e61b96</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>cend</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a70aa149a2b100887a04101b0e13b60a7</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_type</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>ac971fd9e588c9f3ae1f946918e435b64</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>empty</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>abfee585dea165b08f8f6a6ede9705eaf</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>iterator</type>
      <name>m_buffer</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a1a293baaa74287518097adf267adea63</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>size_type</type>
      <name>m_size</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a649671853e185e951473c32403369974</anchor>
      <arglist></arglist>
    </member>
    <member kind="friend">
      <type>friend void</type>
      <name>swap</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1span_1_1Span.html</anchorfile>
      <anchor>a61feb2a38a441d5749f4f814022e55bb</anchor>
      <arglist>(Span&lt; T &gt; &amp;x, Span&lt; T &gt; &amp;y)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::Statistics</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</filename>
    <member kind="variable">
      <type>int</type>
      <name>iterations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>a9a3fdf9f19e5eca76e3bcf6c1288e39f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>r_creations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>ab9fe04e2282fd9e6fd3b6e21e948780d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>q_creations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>ab7af8cd1fe8aea3f2f034244bae0dac9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>p_creations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>a0eed3bdf7711c1a5ca19ede06b3ae2ff</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>q_deletions</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>af9d6d357bf7a662cbdd2d8fb7941a8c6</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>d_creations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>a0f4582c7717868ac6f6ab978e5d0eddc</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>best_r_creations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>a50956f6d06d873e65c5bc6075811be50</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>current_r_creations</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>ac884780bacf914ba9caf27b8644590fb</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>line_searches</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>a76ceef341a3bac5e102fb961ffda9fba</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>line_search_steps</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>a9adab89e699fdb3a96e215790683355b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>rq_ops</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>aa8b840a9a2de9f7c11a72811f71a8b6a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>qr_ops</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>abd2e9fa234e44677bb31e1c4bdf20953</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>rr_ops</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>a77ff925049612a22bbd87cde3c834dfd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>qq_ops</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>afc80d997e12454aa6ea3b96e59f433a8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>rp_ops</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>aa6fc3a0ec490183fe0fb285526eae1e4</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>qp_ops</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1Statistics.html</anchorfile>
      <anchor>a8b85920c1e8007df608ca10f7a5be30c</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::util::StringFacet</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1util_1_1StringFacet.html</filename>
    <member kind="function">
      <type>std::string</type>
      <name>toupper</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1util_1_1StringFacet.html</anchorfile>
      <anchor>adc6e91213f9ab8b947d90455e716b2d7</anchor>
      <arglist>(std::string in) const</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>tolower</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1util_1_1StringFacet.html</anchorfile>
      <anchor>ac9a075ac1254aa1e8a684e8e5365afd8</anchor>
      <arglist>(std::string in) const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>tobool</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1util_1_1StringFacet.html</anchorfile>
      <anchor>ac6623fcc4e51079facbef4304f56eeb1</anchor>
      <arglist>(const std::string &amp;in) const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static void</type>
      <name>crop_space</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1util_1_1StringFacet.html</anchorfile>
      <anchor>afbb704cbae4df3d7b2c3d701adf1a4de</anchor>
      <arglist>(std::string &amp;path)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static std::map&lt; std::string, std::string &gt;</type>
      <name>parse_keyval_string</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1util_1_1StringFacet.html</anchorfile>
      <anchor>a8c650586437277b4443f0bea9e9d72fd</anchor>
      <arglist>(std::string s)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::SubspaceSolverLinEig</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</filename>
    <templarg>class RT</templarg>
    <templarg>class QT</templarg>
    <templarg>class PT</templarg>
    <base>molpro::linalg::itsolv::subspace::ISubspaceSolver</base>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a9636bb82eb7bac6787d3208a1b1b3c1c</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a143f11759fe34eb3ff0a24f1ab07fe89</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::R</type>
      <name>R</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>af4e69763c062c73387807e843f004d04</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::Q</type>
      <name>Q</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a718f82082b0c6372bc5372fb74b2e9d3</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::P</type>
      <name>P</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>aa8fcbb8cf857bf8185a82774c6b17205</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SubspaceSolverLinEig</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>ae101302d4d139fe747f4a87dfbaf7696</anchor>
      <arglist>(std::shared_ptr&lt; Logger &gt; logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>ab98147fb341f860edf75145eecec0392</anchor>
      <arglist>(IXSpace&lt; R, Q, P &gt; &amp;xspace, const size_t nroots_max) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_error</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a44a9305e4ee7ff74a2b2378b9e5d755e</anchor>
      <arglist>(int root, value_type_abs error) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_error</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a28c48233a52096fc6185017555ae22ed</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const std::vector&lt; value_type_abs &gt; &amp;errors) override</arglist>
    </member>
    <member kind="function">
      <type>const Matrix&lt; value_type &gt; &amp;</type>
      <name>solutions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>ae3e779ff0db09d996d79546c3e1f6148</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; value_type &gt; &amp;</type>
      <name>eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a7f19feff38aadceb5b73bd5884413e01</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; value_type_abs &gt; &amp;</type>
      <name>errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>ac7350d5d571c9286f9a0f09f05c622ec</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>afaa8f581eee9b0835a1b3ed70a4e478d</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>abcb40eebed55e1c2c61a3d27a06e1885</anchor>
      <arglist>(bool hermitian)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>get_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a661fad2b5b4e05358e6e5983b96b80e1</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_augmented_hessian</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>aa19b812b2c9f2acf9e7fcc67d8842faa</anchor>
      <arglist>(double parameter)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>get_augmented_hessian</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a2ad9fe0b4687a1c69443847939afbe23</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable">
      <type>value_type_abs</type>
      <name>m_svd_solver_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a4e7db02c63a673f14f1b6e6a9dbd3697</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>solve_eigenvalue</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a4846dfc79950e336bac733d8bcd148eb</anchor>
      <arglist>(IXSpace&lt; R, Q, P &gt; &amp;xspace, const size_t nroots_max)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>solve_linear_equations</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>ae16062f999c44d5cac16953d460335a4</anchor>
      <arglist>(IXSpace&lt; R, Q, P &gt; &amp;xspace)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Matrix&lt; value_type &gt;</type>
      <name>m_solutions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a7a7b63e625561fcd2a4b84a7404681e3</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; value_type &gt;</type>
      <name>m_eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a61ec965600868429b84817938e838106</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; value_type_abs &gt;</type>
      <name>m_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a07d481f0c6b7e113268edb9a41a772b5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>m_logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>a7be7cf20dad5492ad492d4741ee92f1d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_hermitian</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>af8c2e3336ad3e1a73a1e378d453f870f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>double</type>
      <name>m_augmented_hessian</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverLinEig.html</anchorfile>
      <anchor>ade5874a5d05f57487f8c1baa2f0b321f</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::SubspaceSolverOptBFGS</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</filename>
    <templarg>class RT</templarg>
    <templarg>class QT</templarg>
    <templarg>class PT</templarg>
    <base>molpro::linalg::itsolv::subspace::ISubspaceSolver</base>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a33feeee2c6467ce392f1571a1988950c</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a6d66ced0ad3b711e0f56c3bb315633bd</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::R</type>
      <name>R</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a0f07928cd826e70216eb6e7ef66525af</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::Q</type>
      <name>Q</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a1777952c7966ba93c91b7917f42f1179</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::P</type>
      <name>P</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>ad31707405e310259033fb777b6410d79</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SubspaceSolverOptBFGS</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>aa29dda1b236aafca5b78f9789d289461</anchor>
      <arglist>(std::shared_ptr&lt; Logger &gt; logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a5e1e046e765127193cf71b1bb45093f2</anchor>
      <arglist>(IXSpace&lt; R, Q, P &gt; &amp;xspace, const size_t nroots_max) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_error</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>ac2ebdf74d36a67d1c028e7d7879e2f6a</anchor>
      <arglist>(int root, value_type_abs error) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_error</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a719c5da3aff3309e03b01d31c03269b4</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const std::vector&lt; value_type_abs &gt; &amp;errors) override</arglist>
    </member>
    <member kind="function">
      <type>const Matrix&lt; value_type &gt; &amp;</type>
      <name>solutions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>ac6dbde31d4bce2d0d8326dfc99bce997</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; value_type &gt; &amp;</type>
      <name>eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a4f91ff19919530ef6ade081a047bd63c</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; value_type_abs &gt; &amp;</type>
      <name>errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>aee72fc161a9d35754814919a8a4c5cca</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>aba323c67f984e9bcb6ebd507812fffa3</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Matrix&lt; value_type &gt;</type>
      <name>m_solutions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a8c414d9dea354ab6413bdb2145704693</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; value_type_abs &gt;</type>
      <name>m_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a70d9ac45cad2ab211409a5453058e23d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>m_logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptBFGS.html</anchorfile>
      <anchor>a04892db61af31dca67e32cb5f22179e7</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::SubspaceSolverOptSD</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</filename>
    <templarg>class RT</templarg>
    <templarg>class QT</templarg>
    <templarg>class PT</templarg>
    <base>molpro::linalg::itsolv::subspace::ISubspaceSolver</base>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a839d68b3c3ab59075c183999d1ca9048</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a8164d3e09afe2c1e10eff576f6d13d2e</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::R</type>
      <name>R</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a96ba8bd9c089a788a66f4c5b9b4217cb</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::Q</type>
      <name>Q</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a16cf2e0e402f59bc8260ea1bb9c4aafb</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::P</type>
      <name>P</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a4fd8815d103d7464933258ff3716072c</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SubspaceSolverOptSD</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a9db628f2e8a335277ed88c45c09450b7</anchor>
      <arglist>(std::shared_ptr&lt; Logger &gt; logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a8f96a3c648e91c6ce61ece635a11fc45</anchor>
      <arglist>(IXSpace&lt; R, Q, P &gt; &amp;xspace, const size_t nroots_max) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_error</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a58cb600cd3b90195ac9092fb71af1acc</anchor>
      <arglist>(int root, value_type_abs error) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_error</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a3a431a73785c10fffd8855822482ccd0</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const std::vector&lt; value_type_abs &gt; &amp;errors) override</arglist>
    </member>
    <member kind="function">
      <type>const Matrix&lt; value_type &gt; &amp;</type>
      <name>solutions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>adfe554aa8183136b8863cedbf43f920a</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; value_type &gt; &amp;</type>
      <name>eigenvalues</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>afe8e41b0f6a305366e87ec3afc732594</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; value_type_abs &gt; &amp;</type>
      <name>errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a5acedd1dd88000f0bee1f614a0c0afa4</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>size</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>ab3594645f031a4c20f80f532e303773b</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="variable">
      <type>value_type_abs</type>
      <name>m_svd_solver_threshold</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a5b96c2cdb8bbc006c0f1ba8a9447b2dd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Matrix&lt; value_type &gt;</type>
      <name>m_solutions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a9e0b48e7bb794eadf645467564bf3b87</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; value_type_abs &gt;</type>
      <name>m_errors</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>ac16df11a100701ec3b689d020581e6ba</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>m_logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverOptSD.html</anchorfile>
      <anchor>a301e1b66d01143e93301ab5317e255e5</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::SubspaceSolverRSPT</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverRSPT.html</filename>
    <templarg>class RT</templarg>
    <templarg>class QT</templarg>
    <templarg>class PT</templarg>
    <base>molpro::linalg::itsolv::subspace::SubspaceSolverLinEig</base>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::value_type</type>
      <name>value_type</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverRSPT.html</anchorfile>
      <anchor>a9557a7ccd184a3532560681d383f0923</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::value_type_abs</type>
      <name>value_type_abs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverRSPT.html</anchorfile>
      <anchor>a431572a085818065f5ba20668454fd12</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::R</type>
      <name>R</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverRSPT.html</anchorfile>
      <anchor>a54f40ab117132ebde311efc8d406d8f4</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::Q</type>
      <name>Q</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverRSPT.html</anchorfile>
      <anchor>a1597fd2c8b9f92e998cf6a57afed1d6b</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename ISubspaceSolver&lt; RT, QT, PT &gt;::P</type>
      <name>P</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverRSPT.html</anchorfile>
      <anchor>a7b1a72e28e82f7fc93edc4fa89b68492</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SubspaceSolverRSPT</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverRSPT.html</anchorfile>
      <anchor>a0634d6654c494d6717e59d0cd80b169c</anchor>
      <arglist>(std::shared_ptr&lt; Logger &gt; logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solve</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1SubspaceSolverRSPT.html</anchorfile>
      <anchor>a7a5e00b4d721319ffa310f6cbdf74620</anchor>
      <arglist>(IXSpace&lt; R, Q, P &gt; &amp;xspace, const size_t nroots_max) override</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::itsolv::SVD</name>
    <filename>structmolpro_1_1linalg_1_1itsolv_1_1SVD.html</filename>
    <templarg>typename T</templarg>
    <member kind="typedef">
      <type>T</type>
      <name>value_type</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1SVD.html</anchorfile>
      <anchor>a787b6d0a9115e2b6a068fa50c6304ff0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>value_type</type>
      <name>value</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1SVD.html</anchorfile>
      <anchor>a9d16357a1a679676daa80856e0b3e66a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; value_type &gt;</type>
      <name>u</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1SVD.html</anchorfile>
      <anchor>ab0013829641328ce9e5bc21b5af372df</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; value_type &gt;</type>
      <name>v</name>
      <anchorfile>structmolpro_1_1linalg_1_1itsolv_1_1SVD.html</anchorfile>
      <anchor>a63ecf4983f4cf443865c97ae6751a20d</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::array::util::Task</name>
    <filename>classmolpro_1_1linalg_1_1array_1_1util_1_1Task.html</filename>
    <templarg>typename Result</templarg>
    <member kind="function">
      <type></type>
      <name>Task</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Task.html</anchorfile>
      <anchor>a95dedcb6a415a96f70cbb6376e33bfc2</anchor>
      <arglist>(std::future&lt; Result &gt; &amp;&amp;task)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Task</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Task.html</anchorfile>
      <anchor>a0732e18c312f900c477a5b0d4f12bc7c</anchor>
      <arglist>(Task &amp;&amp;other)=default</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~Task</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Task.html</anchorfile>
      <anchor>ae8707b142be02a1a9ea6ac24cae4c4af</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>test</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Task.html</anchorfile>
      <anchor>aef8eb8d724d6d3bf23d5ccde69e0787a</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>Result</type>
      <name>wait</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Task.html</anchorfile>
      <anchor>aac1fc9ecc6188d27bd71a5cd2899eddb</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static Task</type>
      <name>create</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Task.html</anchorfile>
      <anchor>a48c277aecabb4172bbc217282de2524a</anchor>
      <arglist>(Func &amp;&amp;f, Args &amp;&amp;...args)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::future&lt; Result &gt;</type>
      <name>m_task</name>
      <anchorfile>classmolpro_1_1linalg_1_1array_1_1util_1_1Task.html</anchorfile>
      <anchor>ad942cad0c9e31a48b145fc7e75054bce</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>molpro::linalg::array::util::TempHandle</name>
    <filename>structmolpro_1_1linalg_1_1array_1_1util_1_1TempHandle.html</filename>
    <templarg>class HandleType</templarg>
    <member kind="variable" static="yes">
      <type>static std::map&lt; std::string, std::weak_ptr&lt; HandleType &gt; &gt;</type>
      <name>handles</name>
      <anchorfile>structmolpro_1_1linalg_1_1array_1_1util_1_1TempHandle.html</anchorfile>
      <anchor>a63da5f203df8218f905857e3f1eb4916</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>molpro::linalg::itsolv::subspace::XSpace</name>
    <filename>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</filename>
    <templarg>class R</templarg>
    <templarg>class Q</templarg>
    <templarg>class P</templarg>
    <base>IXSpace&lt; R, Q, P &gt;</base>
    <member kind="function">
      <type></type>
      <name>XSpace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>aefe44c3b84f05eea1dc1cd03c10b0bef</anchor>
      <arglist>(const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers, const std::shared_ptr&lt; Logger &gt; &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>update_qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a2dcb242fbc0afca00d2b44f666702932</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;params, const CVecRef&lt; R &gt; &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>update_dspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a40c82457423bf5663ed83a213484e955</anchor>
      <arglist>(VecRef&lt; Q &gt; &amp;params, VecRef&lt; Q &gt; &amp;actions) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>update_pspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a64639a8ee6c512fcc860875eb76cede7</anchor>
      <arglist>(const CVecRef&lt; P &gt; &amp;params, const array::Span&lt; value_type &gt; &amp;pp_action_matrix) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>add_rhs_equations</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a3e5656317e00535ceca0becaa1179fcd</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;rhs)</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>rhs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a702fb6e7fdaac322789caa8b31e6d861</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; value_type_abs &gt; &amp;</type>
      <name>rhs_norm</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>abedea55ea62cd8eae63d25b9900deed9</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const Dimensions &amp;</type>
      <name>dimensions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>aa35c4bab0206253fba9f8f2a9aa2daa5</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>erase</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>aec57d0c249a6abe2c2df69228dd3827d</anchor>
      <arglist>(size_t i) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>eraseq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>abd88486b1a5aceea1d9c1811befd59ba</anchor>
      <arglist>(size_t i) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>erasep</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>ad1df3095682bdb42e4c6d63a2db0d0fe</anchor>
      <arglist>(size_t i) override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>erased</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a515b79e15128de00b559dc76eeb32b07</anchor>
      <arglist>(size_t i) override</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; P &gt;</type>
      <name>paramsp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a9d757386d1af02d10c973984b8ab5e49</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; Q &gt;</type>
      <name>paramsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a8d1bf0139d948557efd047df432c81cc</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; Q &gt;</type>
      <name>actionsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>aaa770ac6e15a8e2a218fa6d504398fa3</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; Q &gt;</type>
      <name>paramsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a9e9d89364b5b02566f6635127a686431</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>VecRef&lt; Q &gt;</type>
      <name>actionsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a5f956af06b59fc44104da37bd2a6051a</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; P &gt;</type>
      <name>paramsp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>aaf57dd6605043709418acf90877bc308</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>paramsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a7e0bb6998b558430bb2837b2c3ac608c</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>actionsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a81d56ef8bf19c9bb355ed7d75069485e</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>paramsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a99ccb72188ce9c164ab5acedeff7b561</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>actionsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>ae0ff6efb10514ff6f5449e12608ff71a</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; P &gt;</type>
      <name>cparamsp</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a19b5feec950fff44b5229ef307298e79</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>cparamsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a01d42ed6ebbf2c25c9e190a12e8cf66f</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>cactionsq</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a150303b6f5b2e34f10c93c1d8e0f8e61</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>cparamsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a2579086ced8a22bc5361ab0839ccd4f4</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>CVecRef&lt; Q &gt;</type>
      <name>cactionsd</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>ae378fcf056b142b82dc6e6decd34e14a</anchor>
      <arglist>() const override</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>af9ba904dede4be6bb48b35329c85fb4a</anchor>
      <arglist>(bool hermitian)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>get_hermiticity</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a0863cf8b61514cd6d4c6bcb691f90df7</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_action_action</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>add12bae4b2859ac50a3f56e35b8b1272</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable">
      <type>PSpace&lt; R, P &gt;</type>
      <name>pspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a6a08f34eb3f54264bcd9f5917e522d22</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>QSpace&lt; R, Q, P &gt;</type>
      <name>qspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>ad4d634f34be831eb4dbf09514069cf8a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>DSpace&lt; Q &gt;</type>
      <name>dspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>aed025f24be0da38d83446600863fcf5d</anchor>
      <arglist></arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>update_dimensions</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a1960d0b465eb5313667fcdd66f6a2a58</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>auto</type>
      <name>update_rhs_with_pspace</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>ac92c8b77befb04daa22114fea51b0453</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>remove_data</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a15bc8a50173852d4d6276602266454ab</anchor>
      <arglist>(size_t i)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt;</type>
      <name>m_handlers</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>ab8df9df05323884e74fd3d172ff67365</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::shared_ptr&lt; Logger &gt;</type>
      <name>m_logger</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a868a058f0de40d594a2682d81d76329e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>Dimensions</type>
      <name>m_dim</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a7ec11686c760067aa9dac90870b78a93</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; Q &gt;</type>
      <name>m_rhs</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a197e910aad5223bd802996fedd7e7376</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>std::vector&lt; value_type_abs &gt;</type>
      <name>m_rhs_norm</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>a64f5c1b582ec8556923a6003d77054aa</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_hermitian</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>aa3d777b87afb66c554abe5853141af59</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>bool</type>
      <name>m_action_dot_action</name>
      <anchorfile>classmolpro_1_1linalg_1_1itsolv_1_1subspace_1_1XSpace.html</anchorfile>
      <anchor>ae3aaee00e6ced3b00c3ac3f68aec4fb1</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>iterative_solver</name>
    <filename>namespaceiterative__solver.html</filename>
    <class kind="interface">iterative_solver::mpi_init</class>
    <class kind="interface">iterative_solver::mpi_rank_global</class>
    <member kind="function">
      <type>subroutine</type>
      <name>test_select</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a762cf435ede70697d8e9df140d410d27</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>integer(kind=mpicomm_kind) function, public</type>
      <name>mpicomm_compute</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a2890fe920d167483bcac382af12afe5d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>set_mpicomm_compute</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a508258aa021cc2d872fba06bd1cff849</anchor>
      <arglist>(comm)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>solve_linear_eigensystem</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a0ec8e7c839d5189981da473f64b78ade</anchor>
      <arglist>(parameters, actions, problem, nroot, generate_initial_guess, max_iter, max_p, thresh, thresh_value, hermitian, verbosity, pname, mpicomm, algorithm, range, options)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>solve_linear_equations</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a0ce14b9eca351507039f752494a53349</anchor>
      <arglist>(parameters, actions, problem, generate_initial_guess, max_iter, max_p, augmented_hessian, thresh, thresh_value, hermitian, verbosity, pname, mpicomm, algorithm, range, options)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>solve_nonlinear_equations</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a05c7eeda78e69ee7c484fe069ab0231b</anchor>
      <arglist>(parameters, actions, problem, nroot, generate_initial_guess, max_iter, thresh, hermitian, verbosity, pname, mpicomm, algorithm, range, options)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>solve_optimization</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>aafc0c445dbb353da38c92aa1ce4640f2</anchor>
      <arglist>(parameters, actions, problem, nroot, generate_initial_guess, max_iter, thresh, thresh_value, hermitian, verbosity, minimize, pname, mpicomm, algorithm, range, options)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>iterative_solver_linear_eigensystem_initialize</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>aeeac19cec453d5df15b17ea1b0f157fb</anchor>
      <arglist>(nq, nroot, thresh, thresh_value, hermitian, verbosity, pname, mpicomm, algorithm, range, options)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>iterative_solver_linear_equations_initialize</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a69c4bfefb260bc5df5c34a84a24a3138</anchor>
      <arglist>(nq, nroot, rhs, augmented_hessian, thresh, thresh_value, hermitian, verbosity, pname, mpicomm, algorithm, range, options)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>iterative_solver_optimize_initialize</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a7ec1c6cf65a2f9d40f2035e7ca0b77ec</anchor>
      <arglist>(nq, thresh, verbosity, minimize, pname, mpicomm, algorithm, range, thresh_value, options)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>iterative_solver_diis_initialize</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a0193308603bbbe44226e4745032d12f7</anchor>
      <arglist>(nq, thresh, verbosity, pname, mpicomm, algorithm, range, options)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>iterative_solver_finalize</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a40a3d04d226f8315ed2a57baa5d78304</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>integer function, dimension(2)</type>
      <name>iterative_solver_range</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a2174400663d4e6e433bf197150e423ce</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>integer function, public</type>
      <name>iterative_solver_add_vector</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a70258968c6cc6e822b9008afcccd8a79</anchor>
      <arglist>(parameters, action, synchronize, value)</arglist>
    </member>
    <member kind="function">
      <type>subroutine</type>
      <name>iterative_solver_add_equations</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>aa8b03a6e82fb1df6df6b25e5017d3714</anchor>
      <arglist>(rhs)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>iterative_solver_solution</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a03f7586355d8dab8557e6cde9fc0078b</anchor>
      <arglist>(roots, parameters, action, synchronize)</arglist>
    </member>
    <member kind="function">
      <type>function, public</type>
      <name>iterative_solver_end_iteration</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>ad609af42ab2050142d98f5ca57522322</anchor>
      <arglist>(solution, residual, synchronize)</arglist>
    </member>
    <member kind="function">
      <type>logical function, public</type>
      <name>iterative_solver_end_iteration_needed</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a5b5146c464788a63c87081ee8f50473d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>integer function, public</type>
      <name>iterative_solver_add_p</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>aa924c3a27ae8d7dfb5f6946c9ab55222</anchor>
      <arglist>(nP, offsets, indices, coefficients, pp, parameters, action, fproc, synchronize)</arglist>
    </member>
    <member kind="function">
      <type>integer function, public</type>
      <name>iterative_solver_suggest_p</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a5165398f200e5ac77b1a80676108cc29</anchor>
      <arglist>(solution, residual, indices, threshold)</arglist>
    </member>
    <member kind="function">
      <type>double precision function, dimension(:), allocatable, public</type>
      <name>iterative_solver_errors</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a8771b451fe4f3381b22d6a64d96f23bd</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>double precision function, dimension(m_nroot), public</type>
      <name>iterative_solver_eigenvalues</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>afad8d58daf37b4b32e1f85046611ece6</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>double precision function, dimension(working_set_size), public</type>
      <name>iterative_solver_working_set_eigenvalues</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a11e3d198ebd7872c9ed08cbbdf9dd012</anchor>
      <arglist>(working_set_size)</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>iterative_solver_solve</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>aeed2ec3b74a21dd29d3ab3853ace408b</anchor>
      <arglist>(parameters, actions, problem, generate_initial_guess, max_iter, max_p)</arglist>
    </member>
    <member kind="function">
      <type>logical function, public</type>
      <name>iterative_solver_converged</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a3e077733271d54ad4d3182d2d9aeb9ab</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>subroutine, public</type>
      <name>apply_p_current_problem</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>af549c18c82c1991ba10874982d745626</anchor>
      <arglist>(p, g, nvec, ranges)</arglist>
    </member>
    <member kind="variable">
      <type>integer, parameter, public</type>
      <name>mpicomm_kind</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>a8e7b0afa0169579bd571c1aac764d4e8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>class(problem_class), pointer, public</type>
      <name>current_problem</name>
      <anchorfile>namespaceiterative__solver.html</anchorfile>
      <anchor>ad1ae0165bd555ba3fbf8d9c6e55d83c0</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>iterative_solver_matrix_problem</name>
    <filename>namespaceiterative__solver__matrix__problem.html</filename>
    <class kind="type">iterative_solver_matrix_problem::matrix_problem</class>
    <member kind="function">
      <type>subroutine</type>
      <name>attach</name>
      <anchorfile>namespaceiterative__solver__matrix__problem.html</anchorfile>
      <anchor>a25d2e7a2cfa8e72dc78666f215609087</anchor>
      <arglist>(this, matrix, RHS)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>iterative_solver_problem</name>
    <filename>namespaceiterative__solver__problem.html</filename>
    <class kind="type">iterative_solver_problem::problem</class>
    <member kind="function">
      <type>logical function</type>
      <name>diagonals</name>
      <anchorfile>namespaceiterative__solver__problem.html</anchorfile>
      <anchor>ad69a0529c3ca725bddee33e94375e6d3</anchor>
      <arglist>(this, d)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>iterative_solver_pspace</name>
    <filename>namespaceiterative__solver__pspace.html</filename>
    <class kind="type">iterative_solver_pspace::pspace</class>
    <member kind="function">
      <type>subroutine</type>
      <name>pspace_ensure</name>
      <anchorfile>namespaceiterative__solver__pspace.html</anchorfile>
      <anchor>a2783f91d094f117bfdabfb21b3d97864</anchor>
      <arglist>(this)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro</name>
    <filename>namespacemolpro.html</filename>
    <namespace>molpro::linalg</namespace>
    <namespace>molpro::profiler</namespace>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg</name>
    <filename>namespacemolpro_1_1linalg.html</filename>
    <namespace>molpro::linalg::array</namespace>
    <namespace>molpro::linalg::iterativesolver</namespace>
    <namespace>molpro::linalg::itsolv</namespace>
    <member kind="function">
      <type>const std::shared_ptr&lt; const molpro::Options &gt;</type>
      <name>options</name>
      <anchorfile>namespacemolpro_1_1linalg.html</anchorfile>
      <anchor>abd04a7feed3604605815d55678fc2369</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>set_options</name>
      <anchorfile>namespacemolpro_1_1linalg.html</anchorfile>
      <anchor>ab848133292d7af277eab1375edea6692</anchor>
      <arglist>(const molpro::Options &amp;options)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::array</name>
    <filename>namespacemolpro_1_1linalg_1_1array.html</filename>
    <namespace>molpro::linalg::array::detail</namespace>
    <namespace>molpro::linalg::array::span</namespace>
    <namespace>molpro::linalg::array::util</namespace>
    <class kind="struct">molpro::linalg::array::array_family</class>
    <class kind="struct">molpro::linalg::array::array_family&lt; T, false, false, true, false &gt;</class>
    <class kind="struct">molpro::linalg::array::array_family&lt; T, false, false, true, true &gt;</class>
    <class kind="struct">molpro::linalg::array::array_family&lt; T, false, true, false, false &gt;</class>
    <class kind="struct">molpro::linalg::array::array_family&lt; T, true, false, false, false &gt;</class>
    <class kind="class">molpro::linalg::array::ArrayHandler</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDDisk</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDDiskDistr</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDDiskSparse</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDDiskSparse&lt; AL, AR, true &gt;</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDefault</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDistr</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDistrDDisk</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDistrSparse</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerDistrSparse&lt; AL, AR, true &gt;</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerIterable</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerIterableSparse</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerIterableSparse&lt; AL, AR, true &gt;</class>
    <class kind="class">molpro::linalg::array::ArrayHandlerSparse</class>
    <class kind="struct">molpro::linalg::array::default_handler</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Distributed, ArrayFamily::Distributed &gt;</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Distributed, ArrayFamily::DistributedDisk &gt;</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Distributed, ArrayFamily::Sparse &gt;</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::DistributedDisk, ArrayFamily::Distributed &gt;</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::DistributedDisk, ArrayFamily::DistributedDisk &gt;</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::DistributedDisk, ArrayFamily::Sparse &gt;</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Iterable, ArrayFamily::Iterable &gt;</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Iterable, ArrayFamily::Sparse &gt;</class>
    <class kind="struct">molpro::linalg::array::default_handler&lt; T, S, ArrayFamily::Sparse, ArrayFamily::Sparse &gt;</class>
    <class kind="class">molpro::linalg::array::DistrArray</class>
    <class kind="class">molpro::linalg::array::DistrArrayDisk</class>
    <class kind="class">molpro::linalg::array::DistrArrayFile</class>
    <class kind="class">molpro::linalg::array::DistrArrayGA</class>
    <class kind="class">molpro::linalg::array::DistrArrayMPI3</class>
    <class kind="class">molpro::linalg::array::DistrArraySpan</class>
    <class kind="struct">molpro::linalg::array::has_mapped_type</class>
    <class kind="struct">molpro::linalg::array::has_mapped_type&lt; A, void_t&lt; typename A::mapped_type &gt; &gt;</class>
    <class kind="struct">molpro::linalg::array::is_disk</class>
    <class kind="struct">molpro::linalg::array::is_disk&lt; T, void_t&lt; typename T::disk_array &gt; &gt;</class>
    <class kind="struct">molpro::linalg::array::is_distributed</class>
    <class kind="struct">molpro::linalg::array::is_distributed&lt; T, void_t&lt; typename T::distributed_array &gt; &gt;</class>
    <class kind="struct">molpro::linalg::array::is_iterable</class>
    <class kind="struct">molpro::linalg::array::is_iterable&lt; T, void_t&lt; decltype(std::begin(std::declval&lt; T &gt;())), decltype(std::end(std::declval&lt; T &gt;())), std::enable_if_t&lt;!is_sparse_v&lt; T &gt; &gt; &gt; &gt;</class>
    <class kind="struct">molpro::linalg::array::mapped_or_value_type</class>
    <class kind="struct">molpro::linalg::array::mapped_or_value_type&lt; A, true &gt;</class>
    <class kind="class">molpro::linalg::array::Span</class>
    <member kind="typedef">
      <type>void</type>
      <name>void_t</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a19258be4ece32e79aa77b39de3132ce4</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename mapped_or_value_type&lt; A &gt;::type</type>
      <name>mapped_or_value_type_t</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a5aeb80119c6ea1e4cd9155b6fb025344</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename default_handler&lt; T, S &gt;::value</type>
      <name>default_handler_t</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a80c986e4a5d6be516962254af8ddb585</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>ArrayFamily</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>acc52e65e7b1ba494644a4783d10a8500</anchor>
      <arglist></arglist>
      <enumvalue file="namespacemolpro_1_1linalg_1_1array.html" anchor="acc52e65e7b1ba494644a4783d10a8500a6adf97f83acf6453d4a6a4b1070f3754">None</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1array.html" anchor="acc52e65e7b1ba494644a4783d10a8500a6b6987a7e21c898769afabc8049cfe3b">Iterable</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1array.html" anchor="acc52e65e7b1ba494644a4783d10a8500a7407fb7e6a4df6392aaabd2368157312">Sparse</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1array.html" anchor="acc52e65e7b1ba494644a4783d10a8500a8c16cbebef45d87fd2b36ce69f46c526">Distributed</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1array.html" anchor="acc52e65e7b1ba494644a4783d10a8500a7767e6bfd66b2c629f74e4be4031e01e">DistributedDisk</enumvalue>
    </member>
    <member kind="function">
      <type>constexpr auto</type>
      <name>check_abs</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a6acd07fa233242b91864374861696f46</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>create_default_handler</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a5890c9a2aa072e5acc1690cb1f1ee95f</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>dot</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a573843fe35b436c5b9671160d6886a18</anchor>
      <arglist>(const DistrArrayDisk &amp;x, const DistrArrayDisk &amp;y)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>dot</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a08a3a46f7c8030c9161938fc76bf7441</anchor>
      <arglist>(const DistrArrayDisk &amp;x, const DistrArray &amp;y)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>dot</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a9127fd56dbfe49e8698f14dcca382295</anchor>
      <arglist>(const DistrArray &amp;x, const DistrArrayDisk &amp;y)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>swap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a5b332a516f38cb8ea680c5ee7656a883</anchor>
      <arglist>(DistrArrayFile &amp;x, DistrArrayFile &amp;y) noexcept</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>swap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>af8b924ed3087a41fa5d4b9627d31b7c0</anchor>
      <arglist>(DistrArraySpan &amp;a1, DistrArraySpan &amp;a2) noexcept</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>swap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>adacc865d1f19cf485d963b55931b55f8</anchor>
      <arglist>(DistrArrayMPI3 &amp;a1, DistrArrayMPI3 &amp;a2) noexcept</arglist>
    </member>
    <member kind="variable">
      <type>constexpr bool</type>
      <name>has_mapped_type_v</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a13f02b59213de0924418fca64e40acf3</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>constexpr bool</type>
      <name>is_sparse_v</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a032b081004cd9ee22ff6d6061eaa7684</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>constexpr bool</type>
      <name>is_iterable_v</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a37f32650280ead1187209883828505be</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>constexpr bool</type>
      <name>is_distributed_v</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>ad3bfb249746efc65f70c9af4fc1667a1</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>constexpr bool</type>
      <name>is_disk_v</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a299c3ce7c7b32979a20409256f6d667b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>constexpr auto</type>
      <name>array_family_v</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>a12b3a57e72d2ca5fd50a2925191f0ff9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::mutex</type>
      <name>s_open_error_mutex</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array.html</anchorfile>
      <anchor>af25a43293d1736bcb38d4f07b8056745</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::array::detail</name>
    <filename>namespacemolpro_1_1linalg_1_1array_1_1detail.html</filename>
    <class kind="struct">molpro::linalg::array::detail::create_default_handler</class>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::array::span</name>
    <filename>namespacemolpro_1_1linalg_1_1array_1_1span.html</filename>
    <class kind="class">molpro::linalg::array::span::Span</class>
    <member kind="function">
      <type>auto</type>
      <name>begin</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1span.html</anchorfile>
      <anchor>a1685994156d0ee2713c481e85594a8d8</anchor>
      <arglist>(Span&lt; T &gt; &amp;x)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>begin</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1span.html</anchorfile>
      <anchor>a10915fd6a2e9dcdac5d6c13e3cce49be</anchor>
      <arglist>(const Span&lt; T &gt; &amp;x)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>end</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1span.html</anchorfile>
      <anchor>a2263f905c14e6ac40e5fd2e372e61006</anchor>
      <arglist>(Span&lt; T &gt; &amp;x)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>end</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1span.html</anchorfile>
      <anchor>a10429ef4c3e04688f04fee388513801b</anchor>
      <arglist>(const Span&lt; T &gt; &amp;x)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::array::util</name>
    <filename>namespacemolpro_1_1linalg_1_1array_1_1util.html</filename>
    <class kind="struct">molpro::linalg::array::util::ArrayHandlerError</class>
    <class kind="class">molpro::linalg::array::util::BufferManager</class>
    <class kind="struct">molpro::linalg::array::util::CompareAbs</class>
    <class kind="class">molpro::linalg::array::util::DistrFlags</class>
    <class kind="class">molpro::linalg::array::util::Distribution</class>
    <class kind="struct">molpro::linalg::array::util::is_std_array</class>
    <class kind="struct">molpro::linalg::array::util::is_std_array&lt; std::array&lt; T, N &gt; &gt;</class>
    <class kind="class">molpro::linalg::array::util::LockMPI3</class>
    <class kind="struct">molpro::linalg::array::util::OperationRegister</class>
    <class kind="struct">molpro::linalg::array::util::RefEqual</class>
    <class kind="class">molpro::linalg::array::util::ScopeLock</class>
    <class kind="class">molpro::linalg::array::util::Task</class>
    <class kind="struct">molpro::linalg::array::util::TempHandle</class>
    <member kind="enumeration">
      <type></type>
      <name>gemm_type</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ac1856c20beb303170f5bbbbf9e08771f</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>inner</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ac1856c20beb303170f5bbbbf9e08771fa6839a240de254a0396bae992eed95b00</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>outer</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ac1856c20beb303170f5bbbbf9e08771fa431bd84fcae17203f58b37dec475624e</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>std::tuple&lt; std::vector&lt; std::tuple&lt; size_t, size_t, size_t &gt; &gt;, std::vector&lt; X &gt;, std::vector&lt; Y &gt;, std::vector&lt; Z &gt; &gt;</type>
      <name>remove_duplicates</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a922d235e8c5336c4ef99bd5f966bb896</anchor>
      <arglist>(const std::list&lt; std::tuple&lt; X, Y, Z &gt; &gt; &amp;reg, EqualX equal_x, EqualY equal_y, EqualZ equal_z)</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; std::pair&lt; DistrArray::index_type, DistrArray::value_type &gt; &gt;</type>
      <name>extrema</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ae6526259fb284a6ef4562a23fd7d2e54</anchor>
      <arglist>(const DistrArray &amp;x, int n)</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; size_t, double &gt;</type>
      <name>select_max_dot_broadcast</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a26c2faa1a3bbc4ec01fea6621b900388</anchor>
      <arglist>(size_t n, std::map&lt; size_t, double &gt; &amp;local_selection, MPI_Comm communicator)</arglist>
    </member>
    <member kind="function">
      <type>Distribution&lt; Ind &gt;</type>
      <name>make_distribution_spread_remainder</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a8a380319734cfdfd55647ef78bd2133d</anchor>
      <arglist>(size_t dimension, int n_chunks)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>select</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ab5a4b5c2cf0b32e0d89fcbe548e239d2</anchor>
      <arglist>(size_t n, const X &amp;x, bool max=false, bool ignore_sign=false)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>select_sparse</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ae3dc31f768fbb5d9b57dea62785f3317</anchor>
      <arglist>(size_t n, const X &amp;x, bool max=false, bool ignore_sign=false)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>select_max_dot</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a657c8079875902dfbf9c0ae4afe9c32a</anchor>
      <arglist>(size_t n, const X &amp;x, const Y &amp;y)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>select_max_dot_iter_sparse</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ace3a48a9dd80f930a8d9f6f2cebaa82a</anchor>
      <arglist>(size_t n, const X &amp;x, const Y &amp;y)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>select_max_dot_sparse</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>acf41fd99e1f0459e85ab4a4289afa84e</anchor>
      <arglist>(size_t n, const X &amp;x, const Y &amp;y)</arglist>
    </member>
    <member kind="function">
      <type>fs::path</type>
      <name>temp_file_name</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>acfea8dacbd30c52cd53641219a873959</anchor>
      <arglist>(const fs::path &amp;base_name, const std::string &amp;suffix)</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; typename array::mapped_or_value_type_t&lt; AL &gt; &gt;</type>
      <name>gemm_inner_distr_distr</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a0c1e6c4b8f6c9b52a60ba4a7e466cdf2</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;yy, const CVecRef&lt; DistrArrayFile &gt; &amp;xx)</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; typename array::mapped_or_value_type_t&lt; AL &gt; &gt;</type>
      <name>gemm_inner_distr_distr</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ab7da757075ac77e8119ce53f2b8fe584</anchor>
      <arglist>(const CVecRef&lt; DistrArrayFile &gt; &amp;xx, const CVecRef&lt; AL &gt; &amp;yy)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer_distr_distr</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a0babf653da3ab28d9a624d08097878f5</anchor>
      <arglist>(const Matrix&lt; typename array::mapped_or_value_type_t&lt; AL &gt; &gt; alphas, const CVecRef&lt; DistrArrayFile &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_distr_distr</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a6d920450e7705203a7618e77b8e16e3a</anchor>
      <arglist>(array::mapped_or_value_type_t&lt; AL &gt; *alphadata, const CVecRef&lt; DistrArrayFile &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy, gemm_type gemm_type)</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; typename array::mapped_or_value_type_t&lt; AL &gt; &gt;</type>
      <name>gemm_inner_distr_distr</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a49ebbbdf6c9acf6f10b00029f2d7de33</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer_distr_distr</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>aeee45304f9ebef815900389e09221b7e</anchor>
      <arglist>(const Matrix&lt; typename array::mapped_or_value_type_t&lt; AL &gt; &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer_distr_sparse</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a783a152fbc37bae2ee82da0ad168c354</anchor>
      <arglist>(const Matrix&lt; typename array::mapped_or_value_type_t&lt; AL &gt; &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy)</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; typename array::mapped_or_value_type_t&lt; AL &gt; &gt;</type>
      <name>gemm_inner_distr_sparse</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a2f6b37985b8983863b5ff745040ba076</anchor>
      <arglist>(const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gemm_outer_default</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>aa499104a489fffa3b3d34c6ae9380551</anchor>
      <arglist>(Handler &amp;handler, const Matrix&lt; typename Handler::value_type &gt; alphas, const CVecRef&lt; AR &gt; &amp;xx, const VecRef&lt; AL &gt; &amp;yy)</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; typename Handler::value_type &gt;</type>
      <name>gemm_inner_default</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>aece0009d0814a09ab5f30da4cb169116</anchor>
      <arglist>(Handler &amp;handler, const CVecRef&lt; AL &gt; &amp;xx, const CVecRef&lt; AR &gt; &amp;yy)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>gather_all</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a55169e39683dee9b69ec1a0c0f6f68c0</anchor>
      <arglist>(const Distribution&lt; size_t &gt; &amp;distr, MPI_Comm commun, double *first_elem)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>swap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ab2eeabbaae45614dfe38954009a36707</anchor>
      <arglist>(DistrFlags &amp;x, DistrFlags &amp;y)</arglist>
    </member>
    <member kind="variable">
      <type>constexpr bool</type>
      <name>is_std_array_v</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a77d2028e5786d2fdb6331300bd4a5471</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>constexpr bool</type>
      <name>is_array_v</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a6ab167ff19207a0141449fb9e359dc7e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>s_temp_file_name_count</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>a3be37e974c75f3bd166cc7303542dd4a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::mutex</type>
      <name>s_mutex</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1array_1_1util.html</anchorfile>
      <anchor>ac5b51a7de2c9b92d70b9342bd48e5ed1</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::iterativesolver</name>
    <filename>namespacemolpro_1_1linalg_1_1iterativesolver.html</filename>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv.html</filename>
    <namespace>molpro::linalg::itsolv::detail</namespace>
    <namespace>molpro::linalg::itsolv::subspace</namespace>
    <namespace>molpro::linalg::itsolv::util</namespace>
    <class kind="class">molpro::linalg::itsolv::ArrayHandlers</class>
    <class kind="struct">molpro::linalg::itsolv::CastOptions</class>
    <class kind="struct">molpro::linalg::itsolv::decay</class>
    <class kind="struct">molpro::linalg::itsolv::decay&lt; std::reference_wrapper&lt; T &gt; &gt;</class>
    <class kind="struct">molpro::linalg::itsolv::has_iterator</class>
    <class kind="class">molpro::linalg::itsolv::Interpolate</class>
    <class kind="struct">molpro::linalg::itsolv::is_complex</class>
    <class kind="struct">molpro::linalg::itsolv::is_complex&lt; std::complex&lt; T &gt; &gt;</class>
    <class kind="class">molpro::linalg::itsolv::IterativeSolver</class>
    <class kind="class">molpro::linalg::itsolv::IterativeSolverTemplate</class>
    <class kind="class">molpro::linalg::itsolv::LinearEigensystem</class>
    <class kind="class">molpro::linalg::itsolv::LinearEigensystemDavidson</class>
    <class kind="struct">molpro::linalg::itsolv::LinearEigensystemDavidsonOptions</class>
    <class kind="struct">molpro::linalg::itsolv::LinearEigensystemOptions</class>
    <class kind="class">molpro::linalg::itsolv::LinearEigensystemRSPT</class>
    <class kind="struct">molpro::linalg::itsolv::LinearEigensystemRSPTOptions</class>
    <class kind="class">molpro::linalg::itsolv::LinearEquations</class>
    <class kind="class">molpro::linalg::itsolv::LinearEquationsDavidson</class>
    <class kind="struct">molpro::linalg::itsolv::LinearEquationsDavidsonOptions</class>
    <class kind="struct">molpro::linalg::itsolv::LinearEquationsOptions</class>
    <class kind="struct">molpro::linalg::itsolv::Logger</class>
    <class kind="class">molpro::linalg::itsolv::NonLinearEquations</class>
    <class kind="struct">molpro::linalg::itsolv::NonLinearEquationsDIISOptions</class>
    <class kind="struct">molpro::linalg::itsolv::NonLinearEquationsOptions</class>
    <class kind="class">molpro::linalg::itsolv::Optimize</class>
    <class kind="class">molpro::linalg::itsolv::OptimizeBFGS</class>
    <class kind="struct">molpro::linalg::itsolv::OptimizeBFGSOptions</class>
    <class kind="struct">molpro::linalg::itsolv::OptimizeOptions</class>
    <class kind="class">molpro::linalg::itsolv::OptimizeSD</class>
    <class kind="struct">molpro::linalg::itsolv::OptimizeSDOptions</class>
    <class kind="struct">molpro::linalg::itsolv::Options</class>
    <class kind="class">molpro::linalg::itsolv::Problem</class>
    <class kind="class">molpro::linalg::itsolv::SolverFactory</class>
    <class kind="struct">molpro::linalg::itsolv::Statistics</class>
    <class kind="struct">molpro::linalg::itsolv::SVD</class>
    <member kind="typedef">
      <type>std::vector&lt; std::reference_wrapper&lt; A &gt; &gt;</type>
      <name>VecRef</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ab44c7c7dfce88d39d260d1f5c7b70fab</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::vector&lt; std::reference_wrapper&lt; const A &gt; &gt;</type>
      <name>CVecRef</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aafd1821a97ea122d56be347e38e90a89</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>typename decay&lt; T &gt;::type</type>
      <name>decay_t</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a2541a0dfd2575a25a8d4976899b63e1d</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::map&lt; std::string, std::string &gt;</type>
      <name>options_map</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a784351a2599ca33773a9809818edb5a1</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>Verbosity</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ac8246e59286ac3630c4c80231d6ed172</anchor>
      <arglist></arglist>
      <enumvalue file="namespacemolpro_1_1linalg_1_1itsolv.html" anchor="ac8246e59286ac3630c4c80231d6ed172a6adf97f83acf6453d4a6a4b1070f3754">None</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1itsolv.html" anchor="ac8246e59286ac3630c4c80231d6ed172a290612199861c31d1036b185b4e69b75">Summary</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1itsolv.html" anchor="ac8246e59286ac3630c4c80231d6ed172a86c1e32c05338b57578313d8a6fa892d">Iteration</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1itsolv.html" anchor="ac8246e59286ac3630c4c80231d6ed172a4042fcadbe61a3300451157e2c9fe651">Detailed</enumvalue>
    </member>
    <member kind="function">
      <type>int</type>
      <name>eigensolver_lapacke_dsyev</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ac08926114067c3eafad9a758df73dabc</anchor>
      <arglist>(const std::vector&lt; double &gt; &amp;matrix, std::vector&lt; double &gt; &amp;eigenvectors, std::vector&lt; double &gt; &amp;eigenvalues, const size_t dimension)</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; SVD&lt; double &gt; &gt;</type>
      <name>eigensolver_lapacke_dsyev</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a3bdd3fd38d1ba9aa765db3684fb866ff</anchor>
      <arglist>(size_t dimension, std::vector&lt; double &gt; &amp;matrix)</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; SVD&lt; double &gt; &gt;</type>
      <name>eigensolver_lapacke_dsyev</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a516f42ae956054880e6649b626657478</anchor>
      <arglist>(size_t dimension, const molpro::linalg::array::span::Span&lt; double &gt; &amp;matrix)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>get_rank</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a9f4b69d5700c5c0199c7424183b6641f</anchor>
      <arglist>(std::vector&lt; value_type &gt; eigenvalues, value_type threshold)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>get_rank</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a3e088fb3da611068c3c17c6335712499</anchor>
      <arglist>(std::list&lt; SVD&lt; value_type &gt; &gt; svd_system, value_type threshold)</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; SVD&lt; value_type &gt; &gt;</type>
      <name>svd_system</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a7fcb72d6b1c715ed5d87f93aabf65581</anchor>
      <arglist>(size_t nrows, size_t ncols, const array::Span&lt; value_type &gt; &amp;m, double threshold, bool hermitian=false, bool reduce_to_rank=false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>printMatrix</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ae7c8c4790146512e2ce34c7c45432411</anchor>
      <arglist>(const std::vector&lt; value_type &gt; &amp;, size_t rows, size_t cols, std::string title=&quot;&quot;, std::ostream &amp;s=molpro::cout)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>eigenproblem</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ab208f9a750c62cbce299278984d87118</anchor>
      <arglist>(std::vector&lt; value_type &gt; &amp;eigenvectors, std::vector&lt; value_type &gt; &amp;eigenvalues, const std::vector&lt; value_type &gt; &amp;matrix, const std::vector&lt; value_type &gt; &amp;metric, size_t dimension, bool hermitian, double svdThreshold, int verbosity, bool condone_complex)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solve_LinearEquations</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a8ef28cdc50441ca2668d749086a6805d</anchor>
      <arglist>(std::vector&lt; value_type &gt; &amp;solution, std::vector&lt; value_type &gt; &amp;eigenvalues, const std::vector&lt; value_type &gt; &amp;matrix, const std::vector&lt; value_type &gt; &amp;metric, const std::vector&lt; value_type &gt; &amp;rhs, size_t dimension, size_t nroot, double augmented_hessian, double svdThreshold, int verbosity)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>solve_DIIS</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aa96fd396113d6611510fc4d99c26d706</anchor>
      <arglist>(std::vector&lt; value_type &gt; &amp;solution, const std::vector&lt; value_type &gt; &amp;matrix, size_t dimension, double svdThreshold, int verbosity=0)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>printMatrix&lt; double &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a42ef971868a57bfead245c92f65cd675</anchor>
      <arglist>(const std::vector&lt; double &gt; &amp;, size_t rows, size_t cols, std::string title, std::ostream &amp;s)</arglist>
    </member>
    <member kind="function">
      <type>template std::list&lt; SVD&lt; double &gt; &gt;</type>
      <name>svd_system</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ad801a9669a2152c41425d26643c2ef95</anchor>
      <arglist>(size_t nrows, size_t ncols, const array::Span&lt; double &gt; &amp;m, double threshold, bool hermitian, bool reduce_to_rank)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>eigenproblem&lt; double &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a4d40717e8aef489ee43e450fa06ee04b</anchor>
      <arglist>(std::vector&lt; double &gt; &amp;eigenvectors, std::vector&lt; double &gt; &amp;eigenvalues, const std::vector&lt; double &gt; &amp;matrix, const std::vector&lt; double &gt; &amp;metric, const size_t dimension, bool hermitian, double svdThreshold, int verbosity, bool condone_complex)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>solve_LinearEquations&lt; double &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aa62cc2cf48964869a1ef3a69c067d23d</anchor>
      <arglist>(std::vector&lt; double &gt; &amp;solution, std::vector&lt; double &gt; &amp;eigenvalues, const std::vector&lt; double &gt; &amp;matrix, const std::vector&lt; double &gt; &amp;metric, const std::vector&lt; double &gt; &amp;rhs, size_t dimension, size_t nroot, double augmented_hessian, double svdThreshold, int verbosity)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>solve_DIIS&lt; double &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a6e048db156aa54f2ee42f5cd169f154f</anchor>
      <arglist>(std::vector&lt; double &gt; &amp;solution, const std::vector&lt; double &gt; &amp;matrix, const size_t dimension, double svdThreshold, int verbosity)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>printMatrix&lt; std::complex&lt; double &gt; &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a07c7f45b484582e6b6b1e6544aadbc2a</anchor>
      <arglist>(const std::vector&lt; std::complex&lt; double &gt; &gt; &amp;, size_t rows, size_t cols, std::string title, std::ostream &amp;s)</arglist>
    </member>
    <member kind="function">
      <type>template std::list&lt; SVD&lt; std::complex&lt; double &gt; &gt; &gt;</type>
      <name>svd_system</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ae6eeb8097d5224e6cca0ff9947dc8af5</anchor>
      <arglist>(size_t nrows, size_t ncols, const array::Span&lt; std::complex&lt; double &gt; &gt; &amp;m, double threshold, bool hermitian, bool reduce_to_rank)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>eigenproblem&lt; std::complex&lt; double &gt; &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aae626e7849fb722b8333eb05abf8b9e3</anchor>
      <arglist>(std::vector&lt; std::complex&lt; double &gt; &gt; &amp;eigenvectors, std::vector&lt; std::complex&lt; double &gt; &gt; &amp;eigenvalues, const std::vector&lt; std::complex&lt; double &gt; &gt; &amp;matrix, const std::vector&lt; std::complex&lt; double &gt; &gt; &amp;metric, const size_t dimension, bool hermitian, double svdThreshold, int verbosity, bool condone_complex)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>solve_LinearEquations&lt; std::complex&lt; double &gt; &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a7f89d13c96f8c037fb2af906e612b5ec</anchor>
      <arglist>(std::vector&lt; std::complex&lt; double &gt; &gt; &amp;solution, std::vector&lt; std::complex&lt; double &gt; &gt; &amp;eigenvalues, const std::vector&lt; std::complex&lt; double &gt; &gt; &amp;matrix, const std::vector&lt; std::complex&lt; double &gt; &gt; &amp;metric, const std::vector&lt; std::complex&lt; double &gt; &gt; &amp;rhs, size_t dimension, size_t nroot, double augmented_hessian, double svdThreshold, int verbosity)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>solve_DIIS&lt; std::complex&lt; double &gt; &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>af57298c8ffb7301296ceb75fb179eab6</anchor>
      <arglist>(std::vector&lt; std::complex&lt; double &gt; &gt; &amp;solution, const std::vector&lt; std::complex&lt; double &gt; &gt; &amp;matrix, const size_t dimension, double svdThreshold, int verbosity)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>read_handler_counts</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aff594b07a52dd297a4fc83b5a4cca785</anchor>
      <arglist>(std::shared_ptr&lt; Statistics &gt; stats, std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; handlers)</arglist>
    </member>
    <member kind="function">
      <type>std::ostream &amp;</type>
      <name>operator&lt;&lt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a2ddf9d9d9e8eb84510e54c64d86dad16</anchor>
      <arglist>(std::ostream &amp;o, const Statistics &amp;statistics)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>precondition_default</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a8229dae1b2c144af940193d4aae8a460</anchor>
      <arglist>(const VecRef&lt; T &gt; &amp;action, const std::vector&lt; double &gt; &amp;shift, const T &amp;diagonals)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>precondition_default</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aa196d0f83c9368c7afbb70fec5609c84</anchor>
      <arglist>(const VecRef&lt; T &gt; &amp;action, const std::vector&lt; double &gt; &amp;shift, const T &amp;diagonals, typename T::iterator *=nullptr)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>precondition_default</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a87291e93a2c4e59b1af5b46cdb7d2db6</anchor>
      <arglist>(const VecRef&lt; T &gt; &amp;action, const std::vector&lt; double &gt; &amp;shift, const T &amp;diagonals, typename std::enable_if&lt;!has_iterator&lt; T &gt;::value, void * &gt;::type=nullptr)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>wrap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a7d650904c9a8f16d1ad6b8f2d3b766d5</anchor>
      <arglist>(ForwardIt begin, ForwardIt end)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>const_cast_wrap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ab2d191cfa0553d457a8277619c82f930</anchor>
      <arglist>(ForwardIt begin, ForwardIt end)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>cwrap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a15fcd8807dbc5342d69f6be91d7a17b5</anchor>
      <arglist>(ForwardIt begin, ForwardIt end)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>wrap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a7ad9256532df6a4b245a3679328edc1d</anchor>
      <arglist>(const IterableContainer &amp;parameters)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>wrap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ae37643fbaca3b182e357855888611778</anchor>
      <arglist>(IterableContainer &amp;parameters)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>const_cast_wrap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a81d731957693dbb73bc153797a651c8f</anchor>
      <arglist>(IterableContainer &amp;parameters)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>cwrap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a676d330a22092146b7c63d7f60170973</anchor>
      <arglist>(IterableContainer &amp;parameters)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>wrap_arg</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a191a03b43f634a6c470a5858842f3760</anchor>
      <arglist>(T &amp;&amp;arg, S &amp;&amp;... args) -&gt; std::enable_if_t&lt; std::conjunction_v&lt; std::is_same&lt; decay_t&lt; T &gt;, decay_t&lt; S &gt; &gt;... &gt;, VecRef&lt; decay_t&lt; T &gt; &gt; &gt;</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>cwrap_arg</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a3c8d471786b9ca7212f93d1636bfd711</anchor>
      <arglist>(T &amp;&amp;arg, S &amp;&amp;... args) -&gt; std::enable_if_t&lt; std::conjunction_v&lt; std::is_same&lt; decay_t&lt; T &gt;, decay_t&lt; S &gt; &gt;... &gt;, CVecRef&lt; decay_t&lt; T &gt; &gt; &gt;</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; size_t &gt;</type>
      <name>find_ref</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ae20486ad3a90d99c4b5d2d24beaac100</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;wparams, ForwardIt begin, ForwardIt end)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; size_t &gt;</type>
      <name>find_ref</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a7e616ad022172235476de18eb323784c</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;wparams, const CVecRef&lt; R &gt; &amp;wparams_ref)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>remove_elements</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aa2245800026088096803c477ba48f674</anchor>
      <arglist>(std::vector&lt; T &gt; params, const std::vector&lt; U &gt; &amp;indices)</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LinearEigensystem&lt; R, Q, P &gt; &gt;</type>
      <name>create_LinearEigensystem</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a616a52291a7ab62bf83431afe0d0b24f</anchor>
      <arglist>(const LinearEigensystemOptions &amp;options, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LinearEigensystem&lt; R, Q, P &gt; &gt;</type>
      <name>create_LinearEigensystem</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a2ad5409d4f13be5dd704ff75ecc9bf67</anchor>
      <arglist>(const std::string &amp;method=&quot;Davidson&quot;, const std::string &amp;options=&quot;&quot;, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LinearEquations&lt; R, Q, P &gt; &gt;</type>
      <name>create_LinearEquations</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>af25f774b165f5eabacfe2ff33a0862ba</anchor>
      <arglist>(const LinearEquationsOptions &amp;options, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; LinearEquations&lt; R, Q, P &gt; &gt;</type>
      <name>create_LinearEquations</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a63f4a8879bb44de7b32fc90729afdf1c</anchor>
      <arglist>(const std::string &amp;method=&quot;Davidson&quot;, const std::string &amp;options=&quot;&quot;, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::tuple&lt; bool, std::unique_ptr&lt; LinearEquations&lt; R, Q, P &gt; &gt; &gt;</type>
      <name>Solve_LinearEquations</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a48c0611afbf73cb378d2124967668ace</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;actions, const Problem&lt; R &gt; &amp;problem, int verbosity=0, bool generate_initial_guess=true, const std::string &amp;method=&quot;Davidson&quot;, const std::string &amp;options=&quot;&quot;, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::tuple&lt; bool, std::unique_ptr&lt; LinearEquations&lt; R, Q, P &gt; &gt; &gt;</type>
      <name>Solve_LinearEquations</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ab88f1f046717dbab4c74ce7758540162</anchor>
      <arglist>(std::vector&lt; R &gt; &amp;parameters, std::vector&lt; R &gt; &amp;actions, const Problem&lt; R &gt; &amp;problem, int verbosity=0, bool generate_initial_guess=true, const std::string &amp;method=&quot;Davidson&quot;, const std::string &amp;options=&quot;&quot;, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>Solve_LinearEquations</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aacae3cfb0660f054b3def5468355b6b9</anchor>
      <arglist>(R &amp;parameters, R &amp;actions, const Problem&lt; R &gt; &amp;problem, int verbosity=0, bool generate_initial_guess=true, const std::string &amp;method=&quot;Davidson&quot;, const std::string &amp;options=&quot;&quot;, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; NonLinearEquations&lt; R, Q, P &gt; &gt;</type>
      <name>create_NonLinearEquations</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aa8102eedebf9255e7ee2e0c9883feda2</anchor>
      <arglist>(const NonLinearEquationsOptions &amp;options=NonLinearEquationsOptions{}, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; NonLinearEquations&lt; R, Q, P &gt; &gt;</type>
      <name>create_NonLinearEquations</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a7a5bc625ccfcaedbc3bae32d4c34cedd</anchor>
      <arglist>(const std::string &amp;method=&quot;DIIS&quot;, const std::string &amp;options=&quot;&quot;, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; Optimize&lt; R, Q, P &gt; &gt;</type>
      <name>create_Optimize</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aa8d07ea5e8b250a8781c29a063083129</anchor>
      <arglist>(const OptimizeOptions &amp;options=OptimizeOptions{}, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; Optimize&lt; R, Q, P &gt; &gt;</type>
      <name>create_Optimize</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a0838aa2a97139616ffe7f037c12bc4a5</anchor>
      <arglist>(const std::string &amp;method=&quot;BFGS&quot;, const std::string &amp;options=&quot;&quot;, const std::shared_ptr&lt; ArrayHandlers&lt; R, Q, P &gt; &gt; &amp;handlers=std::make_shared&lt; molpro::linalg::itsolv::ArrayHandlers&lt; R, Q, P &gt; &gt;())</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ad845d8e72c3518d938b7c0250c70de87</anchor>
      <arglist>(const Interpolate::point &amp;lhs, const Interpolate::point &amp;rhs)</arglist>
    </member>
    <member kind="function">
      <type>std::ostream &amp;</type>
      <name>operator&lt;&lt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a32fc8c3fc3217ba8e8c4abb2c7de9f2e</anchor>
      <arglist>(std::ostream &amp;os, const Interpolate &amp;interpolant)</arglist>
    </member>
    <member kind="function">
      <type>std::ostream &amp;</type>
      <name>operator&lt;&lt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a165abd9e57fd61f69482f77d58c8940f</anchor>
      <arglist>(std::ostream &amp;os, const Interpolate::point &amp;p)</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; SVD&lt; value_type &gt; &gt;</type>
      <name>svd_eigen_jacobi</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>aed5f5152ca0039709fc9f29a60595f68</anchor>
      <arglist>(size_t nrows, size_t ncols, const array::Span&lt; value_type &gt; &amp;m, double threshold)</arglist>
    </member>
    <member kind="function">
      <type>std::list&lt; SVD&lt; value_type &gt; &gt;</type>
      <name>svd_eigen_bdcsvd</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a30d21b580f20c0e56221dfc117418b6a</anchor>
      <arglist>(size_t nrows, size_t ncols, const array::Span&lt; value_type &gt; &amp;m, double threshold)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>printMatrix&lt; value_type &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ae03ed5424809a945ea8755ea935e65a3</anchor>
      <arglist>(const std::vector&lt; value_type &gt; &amp;, size_t rows, size_t cols, std::string title, std::ostream &amp;s)</arglist>
    </member>
    <member kind="function">
      <type>template size_t</type>
      <name>get_rank&lt; value_type &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a7ee90b05dcbb6a130dd3a37de50dabce</anchor>
      <arglist>(std::vector&lt; value_type &gt; eigenvalues, value_type threshold)</arglist>
    </member>
    <member kind="function">
      <type>template std::list&lt; SVD&lt; value_type &gt; &gt;</type>
      <name>svd_system&lt; value_type &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ab7117e36c19daa3142245f1493a5fb8f</anchor>
      <arglist>(size_t nrows, size_t ncols, const array::Span&lt; value_type &gt; &amp;m, double threshold, bool hermitian, bool reduce_to_rank)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>eigenproblem&lt; value_type &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>a6a4c7295faf87794ff06f76866fbe3ca</anchor>
      <arglist>(std::vector&lt; value_type &gt; &amp;eigenvectors, std::vector&lt; value_type &gt; &amp;eigenvalues, const std::vector&lt; value_type &gt; &amp;matrix, const std::vector&lt; value_type &gt; &amp;metric, size_t dimension, bool hermitian, double svdThreshold, int verbosity, bool condone_complex)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>solve_LinearEquations&lt; value_type &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>ad5346e25e508bbda87664acdfd3b029f</anchor>
      <arglist>(std::vector&lt; value_type &gt; &amp;solution, std::vector&lt; value_type &gt; &amp;eigenvalues, const std::vector&lt; value_type &gt; &amp;matrix, const std::vector&lt; value_type &gt; &amp;metric, const std::vector&lt; value_type &gt; &amp;rhs, size_t dimension, size_t nroot, double augmented_hessian, double svdThreshold, int verbosity)</arglist>
    </member>
    <member kind="function">
      <type>template void</type>
      <name>solve_DIIS&lt; value_type &gt;</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv.html</anchorfile>
      <anchor>abdf2f13f0106989d610fa4b05ce05eee</anchor>
      <arglist>(std::vector&lt; value_type &gt; &amp;solution, const std::vector&lt; value_type &gt; &amp;matrix, size_t dimension, double svdThreshold, int verbosity)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv::detail</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</filename>
    <namespace>molpro::linalg::itsolv::detail::dspace</namespace>
    <class kind="class">molpro::linalg::itsolv::detail::DSpaceResetter</class>
    <member kind="function">
      <type>std::vector&lt; std::pair&lt; size_t, size_t &gt; &gt;</type>
      <name>parameter_batches</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a6d3d8c381bac1eb7fa2ff0752e0c6cea</anchor>
      <arglist>(const size_t nsol, const size_t nparam)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>construct_solution</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a4d779275bdb5444c9ed12bfa01ca58ea</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;params, const std::vector&lt; int &gt; &amp;roots, const subspace::Matrix&lt; double &gt; &amp;solutions, const std::vector&lt; std::reference_wrapper&lt; P &gt; &gt; &amp;pparams, const std::vector&lt; std::reference_wrapper&lt; Q &gt; &gt; &amp;qparams, const std::vector&lt; std::reference_wrapper&lt; Q &gt; &gt; &amp;dparams, size_t oP, size_t oQ, size_t oD, ArrayHandlers&lt; R, Q, P &gt; &amp;handlers)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::vector&lt; T &gt; &gt;</type>
      <name>construct_vectorP</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a3af95639074feb491b36180e8c6ef36c</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;roots, const subspace::Matrix&lt; T &gt; &amp;solutions, const size_t oP, const size_t nP)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>normalise</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a61eb49b7dfb7acbac0b882e500c0e742</anchor>
      <arglist>(const size_t n_roots, const VecRef&lt; R &gt; &amp;params, const VecRef&lt; R &gt; &amp;actions, array::ArrayHandler&lt; R, R &gt; &amp;handler, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>update_errors</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a83f1c4849c0c4a40c4adad952fee9da7</anchor>
      <arglist>(std::vector&lt; T &gt; &amp;errors, const CVecRef&lt; R &gt; &amp;residual, array::ArrayHandler&lt; R, R &gt; &amp;handler)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; int &gt;</type>
      <name>select_working_set</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a6edb428467caf65c2df6d4f0d34a85b8</anchor>
      <arglist>(const size_t nw, const std::vector&lt; T &gt; &amp;errors, const T threshold, const std::vector&lt; T &gt; &amp;value_errors, const T value_threshold)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>normalise</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a3b5b8edb5325e4d909f7dac0a1d2541d</anchor>
      <arglist>(VecRef&lt; R &gt; &amp;params, array::ArrayHandler&lt; R, R &gt; &amp;handler, Logger &amp;logger, double thresh=1.0e-14)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>append_overlap_with_r</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>ad63c6172e8baa3a325c27a4e5314c203</anchor>
      <arglist>(const subspace::Matrix&lt; value_type &gt; &amp;overlap, const CVecRef&lt; R &gt; &amp;params, const CVecRef&lt; P &gt; &amp;pparams, const CVecRef&lt; Q &gt; &amp;qparams, const CVecRef&lt; Q &gt; &amp;dparams, ArrayHandlers&lt; R, Q, P &gt; &amp;handlers, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>limit_qspace_size</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a0112d8e773c6ee2070808a2c4453220c</anchor>
      <arglist>(const subspace::Dimensions &amp;dims, const size_t max_size_qspace, const subspace::Matrix&lt; value_type &gt; &amp;solutions, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>construct_dspace</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>ae45dadab3ea0843302f59275c8654e50</anchor>
      <arglist>(const subspace::Matrix&lt; value_type &gt; &amp;solutions, const subspace::IXSpace&lt; R, Q, P &gt; &amp;xspace, const std::vector&lt; int &gt; &amp;q_delete, const value_type_abs norm_thresh, const value_type_abs svd_thresh, array::ArrayHandler&lt; Q, Q &gt; &amp;handler, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>modified_gram_schmidt</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>ae5d64fe02b3f3b416bb9b172f77f1bf5</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;rparams, const subspace::Matrix&lt; value_type &gt; &amp;overlap, const subspace::Dimensions &amp;dims, const CVecRef&lt; P &gt; &amp;pparams, const CVecRef&lt; Q &gt; &amp;qparams, const CVecRef&lt; Q &gt; &amp;dparams, const value_type_abs norm_thresh, ArrayHandlers&lt; R, Q, P &gt; &amp;handlers, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>redundant_parameters</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>af6f410b0781a5f1d5b8c88afece5d4b1</anchor>
      <arglist>(const subspace::Matrix&lt; value_type &gt; &amp;overlap, const size_t oR, const size_t nR, const value_type_abs svd_thresh, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>get_new_working_set</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>ad3c2afce975327f0d495ef8f7d2eff86</anchor>
      <arglist>(const std::vector&lt; int &gt; &amp;working_set, const CVecRef&lt; R &gt; &amp;params, const CVecRef&lt; R &gt; &amp;wparams)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>propose_rspace</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a712059d4e879816d87991146c30a9166</anchor>
      <arglist>(IterativeSolver&lt; R, Q, P &gt; &amp;solver, const VecRef&lt; R &gt; &amp;parameters, const VecRef&lt; R &gt; &amp;residuals, subspace::IXSpace&lt; R, Q, P &gt; &amp;xspace, subspace::ISubspaceSolver&lt; R, Q, P &gt; &amp;subspace_solver, ArrayHandlers&lt; R, Q, P &gt; &amp;handlers, Logger &amp;logger, value_type_abs svd_thresh, value_type_abs res_norm_thresh, int max_size_qspace, molpro::profiler::Profiler &amp;profiler)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>resize_qspace</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a051628c377dc11c2eabb3beec9900242</anchor>
      <arglist>(subspace::IXSpace&lt; R, Q, P &gt; &amp;xspace, const subspace::Matrix&lt; value_type &gt; &amp;solutions, int m_max_Qsize_after_reset, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>max_overlap_with_R</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail.html</anchorfile>
      <anchor>a7e59cce18c2d88b4a7f69312b4736043</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;rparams, const CVecRef&lt; Q &gt; &amp;qparams, array::ArrayHandler&lt; R, Q &gt; &amp;handler, Logger &amp;logger)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv::detail::dspace</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv_1_1detail_1_1dspace.html</filename>
    <member kind="function">
      <type>auto</type>
      <name>construct_projected_solution</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail_1_1dspace.html</anchorfile>
      <anchor>a7e1b57ecc524a7a58cec557606ed4e79</anchor>
      <arglist>(const subspace::Matrix&lt; value_type &gt; &amp;solutions, const subspace::Dimensions &amp;dims, const std::vector&lt; int &gt; &amp;remove_qspace, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>construct_projected_solutions_overlap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail_1_1dspace.html</anchorfile>
      <anchor>a34cb0567313603e0ded6be29155b9d4e</anchor>
      <arglist>(const subspace::Matrix&lt; value_type &gt; &amp;solutions_proj, const subspace::Matrix&lt; value_type &gt; &amp;overlap, const subspace::Dimensions &amp;dims, const std::vector&lt; int &gt; &amp;remove_qspace, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>remove_null_norm_and_normalise</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail_1_1dspace.html</anchorfile>
      <anchor>a518a05e2a79e322ff35c353644827b5d</anchor>
      <arglist>(subspace::Matrix&lt; value_type &gt; &amp;parameters, subspace::Matrix&lt; value_type &gt; &amp;overlap, const value_type_abs norm_thresh, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>remove_null_projected_solutions</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail_1_1dspace.html</anchorfile>
      <anchor>a4864b4a60dab5a5e7e3821fb31b7aeb8</anchor>
      <arglist>(const subspace::Matrix&lt; value_type &gt; &amp;solutions_proj, const subspace::Matrix&lt; value_type &gt; &amp;overlap_proj, const value_type_abs svd_thresh, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>construct_full_subspace_overlap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1detail_1_1dspace.html</anchorfile>
      <anchor>a7cce8d93e8628a08f5fe029ba5a71dff</anchor>
      <arglist>(const subspace::Matrix&lt; value_type &gt; &amp;solutions_proj, const subspace::Dimensions &amp;dims, const std::vector&lt; int &gt; &amp;remove_qspace, const subspace::Matrix&lt; value_type &gt; &amp;overlap, const size_t nR)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv::subspace</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html</filename>
    <namespace>molpro::linalg::itsolv::subspace::qspace</namespace>
    <namespace>molpro::linalg::itsolv::subspace::util</namespace>
    <namespace>molpro::linalg::itsolv::subspace::xspace</namespace>
    <class kind="struct">molpro::linalg::itsolv::subspace::Dimensions</class>
    <class kind="class">molpro::linalg::itsolv::subspace::DSpace</class>
    <class kind="struct">molpro::linalg::itsolv::subspace::ISubspaceSolver</class>
    <class kind="class">molpro::linalg::itsolv::subspace::IXSpace</class>
    <class kind="class">molpro::linalg::itsolv::subspace::Matrix</class>
    <class kind="class">molpro::linalg::itsolv::subspace::PSpace</class>
    <class kind="struct">molpro::linalg::itsolv::subspace::QSpace</class>
    <class kind="class">molpro::linalg::itsolv::subspace::SubspaceSolverLinEig</class>
    <class kind="class">molpro::linalg::itsolv::subspace::SubspaceSolverOptBFGS</class>
    <class kind="class">molpro::linalg::itsolv::subspace::SubspaceSolverOptSD</class>
    <class kind="class">molpro::linalg::itsolv::subspace::SubspaceSolverRSPT</class>
    <class kind="class">molpro::linalg::itsolv::subspace::XSpace</class>
    <member kind="typedef">
      <type>std::map&lt; EqnData, Matrix&lt; double &gt; &gt;</type>
      <name>SubspaceData</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html</anchorfile>
      <anchor>a307dbd210d2a4c973cc1d902e153f038</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>EqnData</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html</anchorfile>
      <anchor>a30629fcb646a0a617c8b85a098a8b195</anchor>
      <arglist></arglist>
      <enumvalue file="namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html" anchor="a30629fcb646a0a617c8b85a098a8b195ac1d9f50f86825a1a2302ec2449c17196">H</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html" anchor="a30629fcb646a0a617c8b85a098a8b195a5dbc98dcc983a70728bd082d1a47546e">S</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html" anchor="a30629fcb646a0a617c8b85a098a8b195a83ff9f9e3dd7561d3dd91204cf546b7e">rhs</enumvalue>
      <enumvalue file="namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html" anchor="a30629fcb646a0a617c8b85a098a8b195a2063c1608d6e0baf80249c42e2be5804">value</enumvalue>
    </member>
    <member kind="function">
      <type>void</type>
      <name>transpose_copy</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html</anchorfile>
      <anchor>a48f71d3d15b8de9d41b31ed27a679f6f</anchor>
      <arglist>(ML &amp;&amp;ml, const MR &amp;mr)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>as_string</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html</anchorfile>
      <anchor>a30dacede15a934f3b97b759ae1d2a7e3</anchor>
      <arglist>(const Mat &amp;m, int precision=6)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>null_data</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace.html</anchorfile>
      <anchor>a4055315e6c87ae26a04be5f58e066c9d</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv::subspace::qspace</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1qspace.html</filename>
    <class kind="struct">molpro::linalg::itsolv::subspace::qspace::QParam</class>
    <member kind="function">
      <type>auto</type>
      <name>cwrap_params</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1qspace.html</anchorfile>
      <anchor>afee738de72b9ee5c99f56ac51d19883b</anchor>
      <arglist>(ForwardIt begin, ForwardIt end)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>wrap_params</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1qspace.html</anchorfile>
      <anchor>a830060ed87a27cfc4e18beaddb125a7d</anchor>
      <arglist>(ForwardIt begin, ForwardIt end)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv::subspace::util</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</filename>
    <namespace>molpro::linalg::itsolv::subspace::util::detail</namespace>
    <member kind="function">
      <type>std::vector&lt; T &gt;</type>
      <name>gram_schmidt</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</anchorfile>
      <anchor>a2dbc0c6797e895338e7d3a64df942f3c</anchor>
      <arglist>(const Matrix&lt; T &gt; &amp;s, Matrix&lt; T &gt; &amp;l, double norm_thresh=1.0e-14)</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; value_type &gt;</type>
      <name>construct_lin_trans_in_orthogonal_set</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</anchorfile>
      <anchor>a6f6950e94c3f91e7ccb96cea152d2aba</anchor>
      <arglist>(const Matrix&lt; value_type &gt; &amp;overlap, const Matrix&lt; value_type &gt; &amp;lin_trans, const std::vector&lt; value_type_abs &gt; &amp;norm)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>modified_gram_schmidt</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</anchorfile>
      <anchor>a84fae4b760805b4b3e0b31eb13fb0ce2</anchor>
      <arglist>(VecRef&lt; R &gt; &amp;params, array::ArrayHandler&lt; R, R &gt; &amp;handler, value_type_abs null_thresh)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>overlap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</anchorfile>
      <anchor>a04cae2d6f7418d21ced0a813444fb7a0</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;left, const CVecRef&lt; Q &gt; &amp;right, array::ArrayHandler&lt; Z, W &gt; &amp;handler) -&gt; std::enable_if_t&lt; detail::Z_and_W_are_one_of_R_and_Q&lt; R, Q, Z, W &gt;, Matrix&lt; double &gt; &gt;</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; double &gt;</type>
      <name>overlap</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</anchorfile>
      <anchor>a7df435b5a278253a9a995488599d622a</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;params, array::ArrayHandler&lt; R, R &gt; &amp;handler)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>matrix_symmetrize</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</anchorfile>
      <anchor>acca5cd45efff25ad75678b06382128e7</anchor>
      <arglist>(Matrix&lt; T &gt; &amp;mat)</arglist>
    </member>
    <member kind="function">
      <type>Matrix&lt; T &gt;::coord_type</type>
      <name>max_element_index</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</anchorfile>
      <anchor>a79bbc77614a73eb3bfbfabae065e8ba9</anchor>
      <arglist>(const std::list&lt; size_t &gt; &amp;rows, const std::list&lt; size_t &gt; &amp;cols, const Matrix&lt; T &gt; &amp;mat)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; size_t &gt;</type>
      <name>eye_order</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util.html</anchorfile>
      <anchor>a96296abbb9a7e19d170b8af1628a2685</anchor>
      <arglist>(const Slice &amp;mat)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv::subspace::util::detail</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail.html</filename>
    <class kind="struct">molpro::linalg::itsolv::subspace::util::detail::is_one_of</class>
    <class kind="struct">molpro::linalg::itsolv::subspace::util::detail::is_one_of&lt; T1, T2 &gt;</class>
    <class kind="struct">molpro::linalg::itsolv::subspace::util::detail::Overlap</class>
    <class kind="struct">molpro::linalg::itsolv::subspace::util::detail::Overlap&lt; R, Q, Z, W, true, false, false &gt;</class>
    <class kind="struct">molpro::linalg::itsolv::subspace::util::detail::Overlap&lt; R, Q, Z, W, true, true, true &gt;</class>
    <member kind="variable">
      <type>constexpr bool</type>
      <name>Z_and_W_are_one_of_R_and_Q</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1util_1_1detail.html</anchorfile>
      <anchor>aa9cf8a8c6babfd9f59a717f58fd74b85</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv::subspace::xspace</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace.html</filename>
    <class kind="struct">molpro::linalg::itsolv::subspace::xspace::NewData</class>
    <member kind="function">
      <type>auto</type>
      <name>update_qspace_data</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace.html</anchorfile>
      <anchor>a7e056a64c3ab82ab660b6141a53a3481</anchor>
      <arglist>(const CVecRef&lt; R &gt; &amp;params, const CVecRef&lt; R &gt; &amp;actions, const CVecRef&lt; P &gt; &amp;pparams, const CVecRef&lt; Q &gt; &amp;qparams, const CVecRef&lt; Q &gt; &amp;qactions, const CVecRef&lt; Q &gt; &amp;dparams, const CVecRef&lt; Q &gt; &amp;dactions, const CVecRef&lt; Q &gt; &amp;rhs, const Dimensions &amp;dims, ArrayHandlers&lt; R, Q, P &gt; &amp;handlers, Logger &amp;logger, bool hermitian=false, bool action_dot_action=false)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>update_dspace_overlap_data</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace.html</anchorfile>
      <anchor>ac061e9c07544907d0c7dbee4e40812e2</anchor>
      <arglist>(const CVecRef&lt; P &gt; &amp;pparams, const CVecRef&lt; Q &gt; &amp;qparams, const CVecRef&lt; Q &gt; &amp;dparams, const CVecRef&lt; Q &gt; &amp;rhs, array::ArrayHandler&lt; Q, P &gt; &amp;handler_qp, array::ArrayHandler&lt; Q, Q &gt; &amp;handler_qq, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>update_dspace_action_data</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace.html</anchorfile>
      <anchor>a683d076ddc577aecc5301b0621316731</anchor>
      <arglist>(const CVecRef&lt; P &gt; &amp;pparams, const CVecRef&lt; Q &gt; &amp;qparams, const CVecRef&lt; Q &gt; &amp;qactions, const CVecRef&lt; Q &gt; &amp;dparams, const CVecRef&lt; Q &gt; &amp;dactions, array::ArrayHandler&lt; Q, P &gt; &amp;handler_qp, array::ArrayHandler&lt; Q, Q &gt; &amp;handler_qq, Logger &amp;logger)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>copy_dspace_eqn_data</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1subspace_1_1xspace.html</anchorfile>
      <anchor>ae9ce0412f1ae9bf22ee40f809b30904c</anchor>
      <arglist>(const NewData &amp;new_data, SubspaceData &amp;data, const subspace::EqnData e, const Dimensions &amp;dims)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::linalg::itsolv::util</name>
    <filename>namespacemolpro_1_1linalg_1_1itsolv_1_1util.html</filename>
    <class kind="struct">molpro::linalg::itsolv::util::ArrayHandlersError</class>
    <class kind="class">molpro::linalg::itsolv::util::StringFacet</class>
    <member kind="function">
      <type>void</type>
      <name>remove_null_vectors</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1util.html</anchorfile>
      <anchor>a24df5019a548240f08f2d6ae28f6f348</anchor>
      <arglist>(subspace::Matrix&lt; value_type &gt; &amp;lin_trans, std::vector&lt; value_type_abs &gt; &amp;norm, const size_t start, const size_t end, const value_type_abs norm_thresh)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>is_iota</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1util.html</anchorfile>
      <anchor>a5a4f2c1ef2abe57c4594e20de21736ba</anchor>
      <arglist>(ForwardIt it_start, EndIterator it_end, Int value_start)</arglist>
    </member>
    <member kind="function">
      <type>Q</type>
      <name>construct_zeroed_copy</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1util.html</anchorfile>
      <anchor>a4c3019057e313770b3fb9ffe4dfc9b11</anchor>
      <arglist>(const R &amp;param, array::ArrayHandler&lt; Q, R &gt; &amp;handler)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>construct_solutions</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1util.html</anchorfile>
      <anchor>a3ece0e5317f702e7935f70d04887803c</anchor>
      <arglist>(const VecRef&lt; R &gt; &amp;params, const std::vector&lt; int &gt; &amp;roots, const subspace::Matrix&lt; double &gt; &amp;solutions, const CVecRef&lt; P &gt; &amp;pparams, const CVecRef&lt; Q &gt; &amp;qparams, const CVecRef&lt; Q &gt; &amp;dparams, size_t oP, size_t oQ, size_t oD, array::ArrayHandler&lt; R, R &gt; &amp;handler_rr, array::ArrayHandler&lt; R, P &gt; &amp;handler_rp, array::ArrayHandler&lt; R, Q &gt; &amp;handler_rq)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>delete_parameters</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1util.html</anchorfile>
      <anchor>aacd4867e14db45ec690217f59c7cff99</anchor>
      <arglist>(std::vector&lt; int &gt; indices, Container &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>std::map&lt; std::string, std::string &gt;</type>
      <name>capitalize_keys</name>
      <anchorfile>namespacemolpro_1_1linalg_1_1itsolv_1_1util.html</anchorfile>
      <anchor>a7c7ffb21f5167c22282fa2e82c2c83cc</anchor>
      <arglist>(const options_map &amp;options, const SFacet &amp;facet)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>molpro::profiler</name>
    <filename>namespacemolpro_1_1profiler.html</filename>
  </compound>
  <compound kind="page">
    <name>md____w_iterative_solver_iterative_solver_doc_LibraryDesign</name>
    <title>Library Design</title>
    <filename>md____w_iterative_solver_iterative_solver_doc_LibraryDesign.html</filename>
  </compound>
  <compound kind="page">
    <name>TopicIterativeSolver</name>
    <title>Iterative solvers</title>
    <filename>TopicIterativeSolver.html</filename>
  </compound>
  <compound kind="page">
    <name>TopicDistrArrays</name>
    <title>Distributed arrays</title>
    <filename>TopicDistrArrays.html</filename>
  </compound>
  <compound kind="page">
    <name>TopicArrayHandlers</name>
    <title>Array handlers</title>
    <filename>TopicArrayHandlers.html</filename>
  </compound>
  <compound kind="page">
    <name>TopicVecRef</name>
    <title>Wrappers: VecRef and CVecRef</title>
    <filename>TopicVecRef.html</filename>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>iterative-solver</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md____w_iterative_solver_iterative_solver_README</docanchor>
  </compound>
</tagfile>
