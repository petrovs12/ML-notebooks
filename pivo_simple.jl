using OrdinaryDiffEq
using Plots

"""
 We define a struct to hold parameters for the system.
 Adjust or extend as needed.
"""
struct PIVOParams
    productAdvRate::Float64
    marketCap::Float64
    churnProb::Float64
    initWorkGen::Float64
    recurringWorkPerCustomer::Float64
    repMult::Float64
    acquireSpeed::Float64
    salary::Float64
    recurringFee::Float64
    onboardFee::Float64
    productProductivityCap::Float64
    compRate::Float64
end

"""
 ODE system for the PIVO model.

 State vector x = [
   product,             # x[1] ∈ [0,1] ideally
   nCustomersEverTried, # x[2]
   totalLostCustomers,  # x[3]
   accumulatedPnL,      # x[4]
   valueAdded           # x[5]
 ]

 We derive:
   totalCurrentCustomers = nCustomersEverTried - totalLostCustomers

 We also define a "reputation" function in terms of totalCurrentCustomers
 and nCustomersEverTried:

   reputation = p.repMult + (1 - p.repMult)* (totalCurrentCustomers)/(nCustomersEverTried + smallEps)

 and the ODEs follow the "story" from the text.
"""
function pivo_ode!(dx, x, p::PIVOParams, t)
    smallEps = 1e-8

    product            = x[1]
    nCustEverTried     = x[2]
    totalLost          = x[3]
    accumPnL           = x[4]
    valAdded           = x[5]

    totalCurrent = nCustEverTried - totalLost

    # Reputation
    rep = p.repMult + (1 - p.repMult) * (totalCurrent)/(nCustEverTried + smallEps)

    # cost to acquire a new customer (some smooth logistic, here just example)
    function σ(z)
        return 1.0 / (1.0 + exp(-z))
    end
    costAcquire = 0.4 * σ( -0.5*(totalCurrent - 4.0) )

    # ODE pieces:

    # 1) d(product)/dt
    #    For simplicity: productAdvRate*(1 - product)*some_work_fraction
    #    We omit the "workOnCToWorkOnPTranslation" for clarity.
    dproduct = p.productAdvRate * (1 - product)

    # 2) new customers come in
    gainCust = p.acquireSpeed * rep * (p.marketCap - nCustEverTried) * product
    losePotential = p.compRate*(p.marketCap - nCustEverTried)
    dnEver = gainCust - losePotential
    # clamp if negative overshoots
    if dnEver < 0 && nCustEverTried <= 0
        dnEver = 0
    end

    # 3) lost customers from "unmet" work
    #    current work = initWorkGen*gainCust + recurringWork*totalCurrent
    currentWork = p.initWorkGen*gainCust + p.recurringWorkPerCustomer*totalCurrent

    #    let's define the rate of adding new value:
    #    d(valAdded)/dt = (product / denom)*some fraction (like 1 for now)
    denom = (1 - product) + (1/p.productProductivityCap)
    dval = (product/denom) * 1.0  # "1.0" means we do all "customer" work

    # unmet work
    unmet = currentWork - dval
    # churn:
    dLost = p.churnProb * max(0, unmet)

    # 4) PnL
    #    We earn "onboardFee" for each new sign-up
    #    We pay costAcquire*salary for each new sign-up
    #    We earn recurringFee * totalCurrent
    #    We pay salary*(some fraction?), but let's keep it simple: just 1.0
    signups = max(gainCust, 0)
    dPnL = p.onboardFee*signups - costAcquire*p.salary*signups +
           p.recurringFee*totalCurrent -
           p.salary*(1.0)

    # fill dx
    dx[1] = dproduct
    dx[2] = dnEver
    dx[3] = dLost
    dx[4] = dPnL
    dx[5] = dval
end

# Example usage:
function main()
    # Initial conditions
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0]  # [product, nEverTried, lost, accumPnL, valAdded]
    time_end=30.0

    # Parameters
    p = PIVOParams(
        0.1,    # productAdvRate
        100.0,   # marketCap
        0.3,    # churnProb
        1.0,    # initWorkGen
        0.5,    # recurringWorkPerCustomer
        0.2,    # repMult
        0.3,    # acquireSpeed
        2.0,    # salary
        0.3,    # recurringFee
        30.0,   # onboardFee
        50.0,    # productProductivityCap
        0.04    # compRate
    )

    # Define ODEProblem
    prob = ODEProblem((du,u,p,t)->pivo_ode!(du,u,p,t), x0, (0.0, time_end), p)

    # Solve
    sol = solve(prob, Tsit5(), abstol=1e-9, reltol=1e-9, saveat=0.5)

    # Plot results
    t = sol.t
    product_vals   = [sol[i][1] for i in 1:length(t)]
    tried_vals     = [sol[i][2] for i in 1:length(t)]
    lost_vals      = [sol[i][3] for i in 1:length(t)]
    pnl_vals       = [sol[i][4] for i in 1:length(t)]
    valadd_vals    = [sol[i][5] for i in 1:length(t)]

    p1 = plot(t, product_vals, label="product", title="product(t)")
    p2 = plot(t, tried_vals,   label="nEverTried", title="nCustomersEverTried(t)")
    p3 = plot(t, lost_vals,    label="lost",  title="totalLostCustomers(t)")
    p4 = plot(t, pnl_vals,     label="PnL",   title="accumPnL(t)")
    p5 = plot(t, valadd_vals,  label="valAdd",title="valueAdded(t)")

    # display them all
    plot!(p1, p2, p3, p4, p5, layout=(3,2), size=(900,600))
end

# Run main if script is executed
println("Starting to solve the ode!()")
main()