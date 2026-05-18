LibraryManager_Append(${PROJECT_NAME}
        SOURCES util.cpp Options.cpp LinearEigensystemDavidsonOptions.cpp LinearEquationsDavidsonOptions.cpp
        NonLinearEquationsDIISOptions.cpp OptimizeBFGSOptions.cpp OptimizeSDOptions.cpp LinearEigensystemRSPTOptions.cpp
        Interpolate.cpp
        PUBLIC_HEADER IterativeSolver.h IterativeSolverTemplate.h LinearEigensystemDavidson.h LinearEigensystemRSPT.h
        LinearEquationsDavidson.h NonLinearEquationsDIIS.h OptimizeBFGS.h OptimizeSD.h Options.h
        Logger.h wrap.h wrap_util.h propose_rspace.h util.h DSpaceResetter.h CastOptions.h
        LinearEigensystemDavidsonOptions.h LinearEigensystemRSPTOptions.h LinearEquationsDavidsonOptions.h
        NonLinearEquationsDIISOptions.h OptimizeBFGSOptions.h OptimizeSDOptions.h
        SolverFactory.h SolverFactory-implementation.h options_map.h
        Interpolate.h
        )
add_subdirectory(subspace)
