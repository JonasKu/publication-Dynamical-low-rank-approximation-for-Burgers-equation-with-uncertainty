include("settings.jl")
include("SGSolver.jl")
include("DLRSolver.jl")
include("plotting.jl")

using NPZ;

close("all");

# initial setup
s = Settings();

s.NCons = 2; # number of conserved basis functions
s.iCons = [1 1; 1 2]; # indices of conserved basis funtions
solver = DLRSolver(s);

PhiQuad = Array(solver.basis.PhiQuad');
PhiQuadCons = Array(solver.basis.PhiQuadCons');
PhiQuadW = Array(solver.basis.PhiQuadW');
fXi = 0.25;

###########################
# run conservative solver #
###########################

X,S,W,uCons = FactorIC(solver);

@time tEnd, Xt,St,Wt,uCons = SolveSplitProjectorSplittingIntegrator(solver,X,S,W,uCons);

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

######################################
# run projector-splitting integrator #
######################################
s.stabilization = 0;
solver = DLRSolver(s);
X,S,W,uCons = FactorIC(solver);

@time tEnd, Xt, St, WDLRPS = SolveProjectorSplittingIntegrator(solver,X,S,W);

uDLR = Array((Xt*St*WDLRPS')')

#################################
# run unconventional integrator #
#################################
s.stabilization = 0;
solver = DLRSolver(s);

@time tEnd, Xt,St,Wt = SolveUnconventionalIntegrator(solver,X,S,W);

uDLRF = Array((Xt*St*Wt')')

#########################
##### Plot solution #####
#########################

plotSolution = Plotting(s,solver.basis,solver.q,s.tEnd);

x0 = 0.42;
PlotInXi(plotSolution,uSG,convert(Int, round(s.Nx*x0/(s.b-s.a))),"Figure4a");
PlotInXi(plotSolution,uDLR,convert(Int, round(s.Nx*x0/(s.b-s.a))),"Figure4b");

Nq = plotSolution.settings.Nq;
Nx = plotSolution.settings.Nx;
NxFine = 1000;
xFine = collect(range(plotSolution.settings.a,plotSolution.settings.b,length=NxFine))
uExact = zeros(NxFine);
varExact = zeros(NxFine);
uPlot = zeros(Nx);
varPlot = zeros(Nx);
vPlot = zeros(Nx);
varVPlot = zeros(Nx);
wPlot = zeros(Nx);
varWPlot = zeros(Nx);
zPlot = zeros(Nx);
varZPlot = zeros(Nx);

#PlotExpectedValue(plotSolution,uSG,uDLR,uDLRF,"noFilter","Figure1");

# start plot
fig = figure("Figure1",figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
for j = 1:Nx
    uVals = EvalAtQuad(plotSolution.basis,uSG[:,j]);
    uPlot[j] = Integral(plotSolution.q,uVals*0.25);
    varPlot[j] = Integral(plotSolution.q,0.25*(uVals.-uPlot[j]).^2);
    vVals = EvalAtQuad(plotSolution.basis,uDLRF[:,j]);
    vPlot[j] = Integral(plotSolution.q,vVals*0.25);
    varVPlot[j] = Integral(plotSolution.q,0.25*(vVals.-vPlot[j]).^2);
    wVals = EvalAtQuad(plotSolution.basis,uDLRCons[:,j]);
    wPlot[j] = Integral(plotSolution.q,wVals*0.25);
    varWPlot[j] = Integral(plotSolution.q,0.25*(wVals.-wPlot[j]).^2);
    zVals = EvalAtQuad(plotSolution.basis,uDLR[:,j]);
    zPlot[j] = Integral(plotSolution.q,zVals*0.25);
    varZPlot[j] = Integral(plotSolution.q,0.25*(zVals.-zPlot[j]).^2);
end
varMax = maximum(varPlot);
expMax = maximum(uPlot);
qFine = Quadrature(200,"Gauss")
exactState = zeros(NxFine,qFine.Nq,qFine.Nq);
for j = 1:NxFine
    for k = 1:qFine.Nq
        for l = 1:qFine.Nq
            exactState[j,k,l] = plotSolution.settings.solutionExact(plotSolution.tEnd,xFine[j],qFine.xi[k],qFine.xi[l])[1];
        end
    end
end
for j = 1:NxFine
    for k = 1:qFine.Nq
        for l = 1:qFine.Nq
            uExact[j] += exactState[j,k,l]*0.25*qFine.w[k]*qFine.w[l];
        end
    end
    for k = 1:qFine.Nq
        for l = 1:qFine.Nq
            varExact[j] += (exactState[j,k,l]-uExact[j])^2 * 0.25*qFine.w[k]*qFine.w[l];
        end
    end
end
ax.plot(plotSolution.x,uPlot, "k--", linewidth=2, label=L"SG$_{100}$", alpha=1.0)
ax.plot(plotSolution.x,zPlot, "g-.", linewidth=2, label=L"DLRA$_{9}, projector-splitting$", alpha=1.0)
ax.plot(plotSolution.x,vPlot, "y-", linewidth=2, label=L"DLRA$_{9}, unconventional$", alpha=1.0)
ax.plot(plotSolution.x,wPlot, "c:", linewidth=2, label=L"DLRA$_{9}$, BC unconventional", alpha=1.0)


ylabel("Expectation", fontsize=20,color="red")
ax.plot(xFine,uExact, "r-", linewidth=2, alpha=0.5)
ax2 = ax[:twinx]() # Create another axis on top of the current axis
ylabel("Standard deviation", fontsize=20,color="blue")
ax2.plot(plotSolution.x,sqrt.(varPlot), "k--", linewidth=2, label="SG", alpha=1.0)
ax2.plot(plotSolution.x,sqrt.(varZPlot), "g-.", linewidth=2, label=L"DLRA$_{9}, PS$", alpha=1.0)
ax2.plot(plotSolution.x,sqrt.(varVPlot), "y-", linewidth=2, label=L"DLRA$_{9}, unconventional$", alpha=1.0)
ax2.plot(plotSolution.x,sqrt.(varWPlot), "c:", linewidth=2, label="unconventional DLRA", alpha=1.0)
#ax2[:set_position](new_position) # Position Method 2
setp(ax2[:get_yticklabels](),color="blue") # Y Axis font formatting
setp(ax[:get_yticklabels](),color="red")
ax2.plot(xFine,sqrt.(varExact), "b-", linewidth=2, alpha=0.5)
#ylimMinus = -0.5;
#ylimPlus = 16.0
#ax[:set_ylim]([ylimMinus,ylimPlus])
ax.set_xlim([plotSolution.settings.a,plotSolution.settings.b])
ax.set_xlabel("x", fontsize=20);
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
ax2.tick_params("both",labelsize=20)
fig.canvas.draw() # Update the figure

###################################################
##                      Plot                     ##
###################################################
#PlotExpectedValue(plotSolution,uSG,uDLR,uDLRF,"Filter","Figure3");

x0 = 0.42;

#PlotInXi(plotSolution,uSG,convert(Int, round(s.Nx*x0/(s.b-s.a))),"Figure4c");
#PlotInXi(plotSolution,uDLR,convert(Int, round(s.Nx*x0/(s.b-s.a))),"Figure4d");
#PlotInXi(plotSolution,uDLRF,convert(Int, round(s.Nx*x0/(s.b-s.a))),"fuDLR");

#################################################
##                  Plot basis                 ##
#################################################
WDLRPlot = zeros(s.r,s.Nq,s.Nq)
xgrid, ygrid = meshgrid(solver.q.xi, 0.5*(1.0.+solver.q.xi))

WQuad = EvalAtQuad(solver.basis,WDLRPS);

for n = 1:s.r
    for i = 1:s.Nq
        for j = 1:s.Nq
            WDLRPlot[n,i,j] = WQuad[(j-1)*s.Nq+i,n]
        end
    end
end

fig = figure("Figure5",figsize=(10,10))
ax = fig.add_subplot(3,3,1, projection="3d")
surf(xgrid, ygrid, WDLRPlot[1,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()

ax = fig.add_subplot(3,3,2, projection="3d")
surf(xgrid, ygrid, WDLRPlot[2,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()

ax = fig.add_subplot(3,3,3, projection="3d")
surf(xgrid, ygrid, WDLRPlot[3,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()

ax = fig.add_subplot(3,3,4, projection="3d")
surf(xgrid, ygrid, WDLRPlot[4,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()

ax = fig.add_subplot(3,3,5, projection="3d")
surf(xgrid, ygrid, WDLRPlot[5,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()

ax = fig.add_subplot(3,3,6, projection="3d")
surf(xgrid, ygrid, WDLRPlot[6,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()

ax = fig.add_subplot(3,3,7, projection="3d")
surf(xgrid, ygrid, WDLRPlot[7,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()

ax = fig.add_subplot(3,3,8, projection="3d")
surf(xgrid, ygrid, WDLRPlot[8,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()

ax = fig.add_subplot(3,3,9, projection="3d")
surf(xgrid, ygrid, WDLRPlot[9,:,:]', cmap=ColorMap("viridis"), alpha=0.7)
tight_layout()