using Pkg

# Remove the problematic packages
try
    Pkg.rm("BoundaryValueDiffEq")
catch e
    println("Error removing BoundaryValueDiffEq: ", e)
end

try
    Pkg.rm("DifferentialEquations")
catch e
    println("Error removing DifferentialEquations: ", e)
end
try
    Pkg.rm("OptimalControl")
catch e
    println("Error removing OptimalControl: ", e)
end
try
    Pkg.rm("Plots")
catch e
    println("Error removing Plots: ", e)
end

# Clear precompilation
try
    Base.compilecache("BoundaryValueDiffEq")
    Base.compilecache("DifferentialEquations")
    Base.compilecache("OptimalControl")
    Base.compilecache("Plots")
catch e
    println("Error clearing precompilation: ", e)
end 

# Add them again
Pkg.add("DifferentialEquations")
Pkg.add("BoundaryValueDiffEq")
Pkg.add("OptimalControl")
Pkg.add("Plots")

Pkg.precompile()