include("settings.jl")
include("SGSolver.jl")
include("DLRSolver.jl")
include("plotting.jl")

using NPZ;

close("all");

s = Settings(1000,30,25);
#s = Settings(600,10,10);
N = s.N;
s.NCons = 0;
s.iCons = 0;

# trun on optimized used of stabilizing terms
optimized = false;

r = [2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 14; 15; 16; 17; 18; 19; 20; 21; 22; 23; 24; 25]
errorExpBack = zeros(length(r));
errorVarBack = zeros(length(r));
errorExpFor = zeros(length(r));
errorVarFor = zeros(length(r));
errorExpSG = zeros(length(r));
errorVarSG = zeros(length(r));

s.useStabilizingTermsS = true;
s.useStabilizingTermsL = true;

s.NCons = 2; # number of conserved basis functions
s.iCons = [1 1; 1 2]; # indices of conserved basis funtions

solver = DLRSolver(s);
PhiQuad = Array(solver.basis.PhiQuad');
PhiQuadCons = Array(solver.basis.PhiQuadCons');
PhiQuadW = Array(solver.basis.PhiQuadW');
fXi = 0.25;
timingsPS = zeros(length(r))
timingsUI = zeros(length(r))
timingsSG = zeros(length(r))

for k = 1:length(r)
    s.r = r[k];
    if optimized
        s.useStabilizingTermsS = false;
        s.useStabilizingTermsL = true;
    end

    solver = DLRSolver(s);
    X,S,W,uCons = FactorIC(solver);

    timingsPS[k] = @elapsed tEnd, X,S,W,uCons = SolveSplitProjectorSplittingIntegrator(solver,X,S,W,uCons);

    # compute moments
    uQ = X*S*W'*PhiQuad + uCons*PhiQuadCons;
    uDLRCons = uQ*Array(solver.basis.PhiQuadWFull')*fXi;
    uDLRCons = Array(uDLRCons')

    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpBack[k], errorVarBack[k] = L2ErrorExpVar(plotSolution,uDLRCons)
    println("rank ",r[k]," : runtime ",timingsPS[k]," ",errorExpBack[k]," ",errorVarBack[k])
end
println("-> DLR projector-splitting integrator DONE.")

s.useStabilizingTermsS = true;
s.useStabilizingTermsL = true;

for k = 1:length(r)
    s.r = r[k];
    if optimized
        s.useStabilizingTermsS = true;
        s.useStabilizingTermsL = false;
    end

    solver = DLRSolver(s);
    X,S,W,uCons = FactorIC(solver);

    timingsUI[k] = @elapsed tEnd, X,S,W,uCons = SolveSplitUnconventionalIntegrator(solver,X,S,W,uCons);
    # compute moments
    uQ = X*S*W'*PhiQuad + uCons*PhiQuadCons;
    uDLRCons = uQ*Array(solver.basis.PhiQuadWFull')*fXi;
    uDLRCons = Array(uDLRCons')

    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpFor[k], errorVarFor[k] = L2ErrorExpVar(plotSolution,uDLRCons)
    println("rank ",r[k]," : runtime ",timingsUI[k]," ",errorExpFor[k]," ",errorVarFor[k])
end
println("-> DLR unconventional integrator DONE.")

s.NCons = 0;
s.iCons = 0;

for k = 1:length(r)
    s.N = r[k];
    if s.N > 5
        #break;
    end
    solver = Solver(s);
    timingsSG[k] = @elapsed tEnd, u = Solve(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpSG[k], errorVarSG[k] = L2ErrorExpVar(plotSolution,u)
    println("moments ",r[k]^2," : runtime ",timingsSG[k]," ",errorExpSG[k]," ",errorVarSG[k])
end
println("-> SG DONE.")

print("Computing SG fine...")
s.N = N;
solver = Solver(s);
timeSGFull = @elapsed tEnd, u = Solve(solver);
plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
errorExpSGFull, errorVarSGFull = L2ErrorExpVar(plotSolution,u)
println(" DONE.")

###### plot results ######

fig = figure("Figure6a",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(r.^2,errorExpSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorExpBack, "g-.<", linewidth=2, label="DLRA, projector-splitting", alpha=1.0)
ax.plot(r,errorExpFor, "m:>", linewidth=2, label="DLRA, unconventional", alpha=1.0)
ax.plot(r,errorExpSGFull*ones(size(r)), "r-", linewidth=2, alpha=0.5)
ylabel("rel. error expectation", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

fig = figure("Figure6b",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(r.^2,errorVarSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorVarBack, "g-.<", linewidth=2, label="DLRA, projector-splitting", alpha=1.0)
ax.plot(r,errorVarFor, "m:>", linewidth=2, label="DLRA, unconventional", alpha=1.0)
ax.plot(r,errorVarSGFull*ones(size(r)), "b-", linewidth=2, alpha=0.5)
ylabel("rel. error variance", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

#### log plot 

fig = figure("Figure6aLog",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(r.^2,errorExpSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorExpBack, "g-.<", linewidth=2, label="DLRA, projector-splitting", alpha=1.0)
ax.plot(r,errorExpFor, "m:>", linewidth=2, label="DLRA, unconventional", alpha=1.0)
ax.plot(r,errorExpSGFull*ones(size(r)), "b-", linewidth=2, alpha=0.5)
ylabel("rel. error expectation", fontsize=20)
xlabel("rank/moments", fontsize=20)
yscale("log")
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

fig = figure("Figure6bLog",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(r.^2,errorVarSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorVarBack, "g-.<", linewidth=2, label="DLRA, projector-splitting", alpha=1.0)
ax.plot(r,errorVarFor, "m:>", linewidth=2, label="DLRA, unconventional", alpha=1.0)
ax.plot(r,errorVarSGFull*ones(size(r)), "r-", linewidth=2, alpha=0.5)
ylabel("rel. error variance", fontsize=20)
xlabel("rank/moments", fontsize=20)
yscale("log")
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

endSG = 17;

# plot timings
fig = figure("Figure7a",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(timingsSG[1:endSG],errorExpSG[1:endSG], "k:o", linewidth=2, label="SG", alpha=1.0, ms = 10)
ax.plot(timingsPS,errorExpBack, "g:<", linewidth=2, label="DLRA, projector-splitting", alpha=1.0, ms = 10)
ax.plot(timingsUI,errorExpFor, "m:>", linewidth=2, label="DLRA, unconventional", alpha=1.0, ms = 10)
#ax.plot(r,errorExpSGFull*ones(size(r)), "r-", linewidth=2, alpha=0.5)
ylabel("rel. error expectation", fontsize=20)
xlabel("runtime [sec]", fontsize=20)
yscale("log")
xscale("log")
grid("on", which="both")
#ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

# plot timings
fig = figure("Figure7b",figsize=(10, 8), dpi=100)
ax = gca()
ax.plot(timingsSG[1:endSG],errorVarSG[1:endSG], "k:o", linewidth=2, label="SG", alpha=1.0, ms = 10)
ax.plot(timingsPS,errorVarBack, "g:<", linewidth=2, label="DLRA, projector-splitting", alpha=1.0, ms = 10)
ax.plot(timingsUI,errorVarFor, "m:>", linewidth=2, label="DLRA, unconventional", alpha=1.0, ms = 10)
#ax.plot(r,errorExpSGFull*ones(size(r)), "r-", linewidth=2, alpha=0.5)
ylabel("rel. error variance", fontsize=20)
xlabel("runtime [sec]", fontsize=20)
yscale("log")
xscale("log")
grid("on", which="both")
#ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure

npzwrite("results/ranksN$(N)Nx$(s.Nx).jld", r)

npzwrite("results/timingsSGN$(N)Nx$(s.Nx).jld", timingsSG)
npzwrite("results/timingsPSN$(N)Nx$(s.Nx).jld", timingsPS)
npzwrite("results/timingsUIN$(N)Nx$(s.Nx).jld", timingsUI)

npzwrite("results/errorExpSGN$(N)Nx$(s.Nx).jld", errorExpSG)
npzwrite("results/errorExpPSN$(N)Nx$(s.Nx).jld", errorExpBack)
npzwrite("results/errorExpUIN$(N)Nx$(s.Nx).jld", errorExpFor)

npzwrite("results/errorVarSGN$(N)Nx$(s.Nx).jld", errorVarSG)
npzwrite("results/errorVarPSN$(N)Nx$(s.Nx).jld", errorVarBack)
npzwrite("results/errorVarUIN$(N)Nx$(s.Nx).jld", errorVarFor)