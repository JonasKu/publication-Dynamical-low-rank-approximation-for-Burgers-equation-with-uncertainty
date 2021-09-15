include("settings.jl")
include("SGSolver.jl")
include("DLRSolver.jl")
include("plotting.jl")

using NPZ;

close("all");

s = Settings();

###########################
# run conservative solver #
###########################
s.NCons = 2; # number of conserved basis functions
s.iCons = [1 1; 1 2]; # indices of conserved basis funtions
solver = DLRSolver(s);

PhiQuad = Array(solver.basis.PhiQuad');
PhiQuadCons = Array(solver.basis.PhiQuadCons');
PhiQuadW = Array(solver.basis.PhiQuadW');
fXi = 0.25;

X,S,W,uCons = FactorIC(solver);

@time tEnd, Xt,St,Wt,uCons = SolveSplitUnconventionalIntegrator(solver,X,S,W,uCons);

# compute moments
uQ = Xt*St*Wt'*PhiQuad + uCons*PhiQuadCons;
uDLRCons = uQ*Array(solver.basis.PhiQuadWFull')*fXi;

uDLRCons = Array(uDLRCons')

##################################
# run stochastic-Galerkin solver #
##################################
s.NCons = 0;
s.iCons = 0;

solver = Solver(s);

@time tEnd, uSG = Solve(solver);

###########################
# run conservative solver #
###########################
s.NCons = 2; # number of conserved basis functions
s.iCons = [1 1; 1 2]; # indices of conserved basis funtions
solver = DLRSolver(s);

@time tEnd, uDLRConsN = SolveNaiveSplitUnconventionalIntegrator(solver);

uDLRConsN = Array(uDLRConsN')

#########################
##### Plot solution #####
#########################
plotSolution = Plotting(s,solver.basis,solver.q,s.tEnd);

PlotExpectedValue(plotSolution,uSG,uDLRCons,uDLRConsN,"noFilter","Figure1");
