- IterativeSolver should be passed ParameterVector object references, and will never work on raw data. The ParameterVector class will take responsibility to provide all linear algebra operations such as Dot, axpy and so on, and anything to do with memory management including paging out to external storage, and anything to do with parallel.
- IterativeSolver is allowed to create arbitrary numbers of ParameterVector objects, but should advise for “most” of them that they do not need to be memory resident. The ParameterVector class can do what it likes with this advice.
- The solver class should provide a simple driving method that has a loop that calls ParameterVector.residual and this->update() or similar in each iteration. However update() should be made public so that a user could write his own loop instead.
On second thoughts, having residual() as a member of ParameterVector is not great, because it means one will need to write a new inheriting class
for each solver type. Redesign this.
- There needs to be a ParameterVectorSet class that is a collection of ParameterVector objects, and which has at least a residual() method which in the default implementation would just call the residual() for each of the ParameterVectors
- Consideration of non-hermitian problems: eigensolution switches to linear equations once the eigenvalue is known to sufficient precision.
- ParameterVector should know whether it is co- or contra-variant, and police this aspect when doing its linear algebra operations - either crash or use its own implementation of the metric when necessary
- std::string status(int level)
- examples
- write Diis and Davidson including examples of use in both solve() and explicit loop mode.
- convergence criterion should normally be g.c but could optionally provide a metric tensor different to the kernel being solved.
- handle errors, including lack of convergency, with try/throw/catch
- tag vectors sent to residual routine with on/off switch - normally all on, but near convergence some may be off, meaning I don't want the residual
- extrapolate vectors other than solution and residual
