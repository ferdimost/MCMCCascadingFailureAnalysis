##
using LinearAlgebra
using LightGraphs
using GraphPlot
using Random
using IterativeSolvers
using Statistics
#using Graphs
using Colors
using Dates
using Plots
using CPUTime
using DelimitedFiles
using CSV, DataFrames
using ColorSchemes
using LaTeXStrings
using StatsBase
using ProgressMeter
using Distributed
using StatsPlots
using Compose
using Cairo
using Fontconfig
using CircularArrays
using Clustering
using SyntheticNetworks
using NetworkLayout
using CairoMakie
using CairoMakie.Colors
using GraphMakie
using JLD
using JLD2
using BenchmarkTools
using SparseArrays
using GeometryBasics
using ColorSchemes: viridis
using Distances
##

##
Base.copy(t::Tuple) = t
#include (".jl")
#include (".jl")
#include (".jl")

""" ((2)) returns selected hour indices though whole data timescale range"""
function create_hour_idx_for_sample(powerdata::String, time::String)::Array{Int64,1}
    Data_Set_OPSD = readdlm(powerdata*".csv", ',')
    timestamps = Data_Set_OPSD[:,1]
    hour_indices = Int[]
    for i in 1:length(timestamps)
        if occursin(time,timestamps[i])
            push!(hour_indices, i)
        end
    end
    return hour_indices #these go in initial_injections function
end

""" ((1)) powerdata is now the reduced data matrix / sample amount is given by number of nodes that network should have"""
function init_injection(powerdata::String, distr_of_node_types::Array{Float64,1}, sample_amount::Int64, PVplant_scaling, hour_indices::Array{Int64,1})::Array{Float64,2} #choose param: hour_select = 1 for "07:00:00" & 2 for "08:00:00" and so on
    Data_Set_OPSD = readdlm(powerdata*".csv", ',')
    #dirstr_of_node_types = e.g. [0,1 für 10% , 0.35 für 35% , etc..]
    #must be 10 entries long array [PV1,PV2,Industry1,Industry2,Industry3,Resident1
    #,Resident3,Resident4,Resident5,Resident6]
    #PV1 -> larger PV plant
    #PV2 -> middle large PV plant
    #Industry1 -> no big pv effect
    #Industry2 -> pv leads to temporary zero import
    #Industry3 -> large import 
    #Resident1 -> no big pv effect
    #resident3 -> in summer net injections
    #Resident4 -> (larger) net injections in summer
    #Resident5 -> just import no PV
    #Resident6 -> same with net injections in summer
    Data_Set_OPSD[2:end,2] = Data_Set_OPSD[2:end,2] * PVplant_scaling #!!after one execution one has to load original powerdata again
    Data_Set_OPSD[2:end,3] = Data_Set_OPSD[2:end,3] * PVplant_scaling
    prelim_distr = distr_of_node_types * (sample_amount-1)
    prelim_distr_round = floor.(prelim_distr)
    #print("prelim_distr_round:", prelim_distr_round)
    while sum(prelim_distr_round) < (sample_amount-1)
        rand_idx = rand(6:length(prelim_distr)) ## fill with random household
        prelim_distr_round[rand_idx] = prelim_distr_round[rand_idx] + 1
    end
    #print("filled_distr:", prelim_distr_round)
    all_nodes_data = Data_Set_OPSD[:,1] #first just the timestamp
    for j in 1:length(prelim_distr_round)    
        for i in 1:prelim_distr_round[j]
            all_nodes_data = hcat(all_nodes_data,Data_Set_OPSD[:,j+1]) #[j+1] skips the timestanmp colum
        end
    end
    all_nodes_data = all_nodes_data[:,2:end] #leaving out the timestamp
    #permute colums
    all_nodes_data_perm_cols = all_nodes_data[:,randperm(length(all_nodes_data[1,:]))]
    
    #all_data = hcat(Data_Set_OPSD[:,1],all_nodes_data_perm_with_slack)
    #slackbus is first introduced in swap!()function
    get_powers_at_hour = all_nodes_data_perm_cols[hour_indices,:]
    get_powers_at_hour = floor.(get_powers_at_hour, digits=3)
    
    return get_powers_at_hour
end
  
"""
load the realtopology graph fro jld file
"""
function load_graph_realtopology(jld_file::String)
    SData = JLD.load(jld_file*".jld")
    adj_matrix = SData["Graph"]
    coordinates = SData["Embedding"]
    results_mod = SData["Modularity_results"]
    adj_matrix, coordinates, results_mod
end
    
"""creates an initial non perturbed injection pattern -> just defined via initial_day..time of day is already specified in P/data"""
function initial_inject_realtopology(P::Array{Float64,2}, init_day::Int64, g, slack_index::Int64)::Array{Float64,1}
    injections = P[init_day,:]
    P_cp = Float64(sum(parse.(BigFloat, string.(injections))))
    Slackbus = sum(P_cp) *  (-1)
    #slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
    insert!(injections,slack_index,Slackbus)
    return injections
end

"""
minimum norm solution is given because B is not invertible 
"""
function flow(g, P::Array{Float64,1})::Array{Float64,1} # calculates the flow of graph g for a certain configuration P
    B = LightGraphs.incidence_matrix(g, Float64, oriented=true) # nxm oriented incidence matrix B
    # node-edge matrix, indexed by [v, i], i is in 1:LightGraphs.ne(g),
    # indexing edge e. Directed graphs: -1 indicates src(e) == v, 1 indicates dst(e) == v.
    # indexing of the edges e does not correspond to sequence of building edges above!!!
    # indexing of edges: https://github.com/JuliaGraphs/LightGraphs.jl/blob/216f5ffa77860d4c39b8d05fe2197d0af75a4241/src/linalg/spectral.jl#L129-L142

    F = lsqr(B, P) # solving BF = P for F by returning minimal norm solution: https://juliamath.github.io/IterativeSolvers.jl/dev/linear_systems/lsqr/
    F_rounded = round.(F; digits = 3)
end

function improved_energy_slacklimit!(g, P::Array{Float64,1}, C::Array{Float64,1}, hh_battery_capacity::Float64, slack_limit::Float64, slack_idx::Int64)
    critical_cases = Tuple{Float64,Int64}[]
    max_momentan_slack_usage = Float64[]
    max_momentan_battery_usage = Float64[]
    B = Array(LightGraphs.incidence_matrix(g, oriented=true))
    C_alt = copy(C)
    P_alt = copy(P)
    g_original = copy(g)
    m = size(B)[2] 
    linefailure_indizes = collect(1:m)
    #slack_edges = get_neighbor_edges_of_slack(g)
    #deleteat!(linefailure_indizes, slack_edges)
    #N = m-length(slack_edges)
    N = m
    G = zeros(N)
    counter_val = 0
    for i in 1:N
        if LightGraphs.ne(g) != 112 || P != P_alt || C != C_alt
            @warn "achtung reset klappt nicht!!"
        end
        g, max_momentan_battery_usage, max_momentan_slack_usage, critical_cases, counter_val = improved_cascading_slacklimit!(g, P, C, hh_battery_capacity, slack_limit, slack_idx, linefailure_indizes[i], max_momentan_battery_usage, max_momentan_slack_usage,critical_cases,counter_val)
        G[i] = biggest_component(g) 
        g = copy(g_original)
        C = copy(C_alt)
        P = copy(P_alt)
    end
    G_av = round(mean(G), digits=3)
    if isempty(max_momentan_battery_usage) == false
        max_bat = round(maximum(max_momentan_battery_usage), digits=3)
        mean_bat = round(mean(max_momentan_battery_usage), digits=3)
    else
        max_bat = 0.0
        mean_bat = 0.0
    end
    if isempty(max_momentan_slack_usage) == false
        max_slack = round(maximum(max_momentan_slack_usage), digits=3)
        mean_slack = round(mean(max_momentan_slack_usage), digits=3)
    else
        max_slack = 0.0
        mean_slack = 0.0
    end
    return G, G_av, max_bat, mean_bat, max_slack, mean_slack, critical_cases, counter_val
end

"""
Calculates the line loadings; given flows and capacities
"""
function eval_line_loadings(flow::Array{Float64,1},C::Array{Float64,1})::Array{Float64,1}
    return abs.(flow)./ C
end

"""
dual swap implements a MCMC step process that alters time and position of injections simultaneously!
Their likelyness to occur are specified by drift_prob and swap_prob parameters.
Else all parameters have occured in other MCMC procedures already.
"""
function dual_swap_new_fixeddrift_realtopology!(dataset::Array{Float64,2}, init_day::Int64, swap::Int64, swap_prob::Float64, drift::Int64, drift_prob::Float64, var::Int, var_at_place::Array{Int64,1}, devi_frac::Float64, step::Int64, slack_index)
    # setting chances for either swapping or drifting
    swap_max = Int(ceil(1/swap_prob))
    swap_chance = rand(1:swap_max)
    drift_old = copy(drift)
    drift = Int(floor(step*drift_prob))
    
    columns = collect(1:length(dataset[1,:]))
    devi_frac_total = Int(ceil(devi_frac * length(dataset[1,:])))
    devi_sample = sample(columns, devi_frac_total, replace = false) #gives elements which get deviation from drift
    #variance
    variations = rand(-var:var,length(devi_sample))
    #for j in 1:length(devi_sample)
    #    var_at_place[devi_sample[j]] = variations[j]
    #end
    circ_dataset = CircularArray(dataset)
   
    # differentiation between four cases
    if drift != drift_old && swap_chance == swap_max
        #print("Drift & Swap")
        swap = swap + 1
        var_at_place = var_at_place .- 1 # -1 to keep the old daily positions of all producers and consumers
        var_at_place[findall(x -> x < -var, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
        
        var_at_place[devi_sample] = variations
        injections = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))] 
    
        N_vertices = length(dataset[1,:])
        A = rand(collect(1:N_vertices))
        B = rand(collect(1:N_vertices))
        while A == B                                 ### <-- Swap procedure
            A = rand(collect(1:N_vertices))
            B = rand(collect(1:N_vertices))
        end
        var_at_place[A], var_at_place[B] = var_at_place[B], var_at_place[A]
        dataset[:,A], dataset[:,B] = dataset[:,B], dataset[:,A]
        injections[A], injections[B] = injections[B], injections[A] 
        
  
    elseif drift != drift_old && swap_chance != swap_max
        #print("Drift")
        swap = swap
        var_at_place = var_at_place .- 1 # -1 to keep the old daily positions of all producers and consumers
        var_at_place[findall(x -> x < -var, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
        
        var_at_place[devi_sample] = variations
        injections = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))] 
        
    elseif drift == drift_old && swap_chance == swap_max
        #print("Swap")
        swap = swap + 1
        drift = drift ##is drift old anyways
        var_at_place[devi_sample] = variations
        injections = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))] 
        
        N_vertices = length(dataset[1,:])
        A = rand(collect(1:N_vertices-1))
        B = rand(collect(1:N_vertices-1))
        while A == B 
            A = rand(collect(1:N_vertices-1))
            B = rand(collect(1:N_vertices-1))
        end
        var_at_place[A], var_at_place[B] = var_at_place[B], var_at_place[A]
        dataset[:,A], dataset[:,B] = dataset[:,B], dataset[:,A]
        injections[A], injections[B] = injections[B], injections[A] 
        
    else
        #print("Nothing Just VAR_AT_Place")
        swap = swap
        var_at_place[devi_sample] = variations
        injections = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
        
    end
    injections = Array{Float64,1}(injections)
    #sum along row of perturbed (by adding the variations) injections *(-1) to get slackbus value 
    P_cp = Float64(sum(parse.(BigFloat, string.(injections))))
    Slackbus = sum(P_cp) *  (-1)
    ###Slackbus = vcat("Slackbus", Slackbus)
    #insert Slackbus column at central position -> nodes_amount/2
    #slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
    insert!(injections,slack_index,Slackbus)
    
    return dataset, injections, drift, swap, var_at_place
end

function probability(ΔE::Float64, T)::Float64 # probability function depends on ΔE and on temperature function
    exp( - ΔE / T) # Boltzmann's constant k_B is set to one
end

function temp_ex00_steps(k,k_max,drift)
    drift_len = (k_max + 1)/ 365
    k = k - drift_len * drift
    0.99 ^ (floor(k/4))   
end

# measure  number of edges gen-gen, con-con, gen-con
function nr_gen_con_realtopology(P::Array{Float64,1}, g)
    gen_gen = 0 # generates empty vector of type 'Any'
    con_con = 0
    gen_con = 0
    for e in collect(LightGraphs.edges(g))
        if P[e.src] > 0 && P[e.dst] > 0
            gen_gen = gen_gen + 1
        elseif P[e.src] < 0 && P[e.dst] < 0
            con_con = con_con + 1
        else
            gen_con = gen_con + 1
        end
    end
    gen_gen, con_con, gen_con
end

"""
    k is the number of relevant edges
    a is a variable list of a quantity
    => gives back the indices of the largest (referring to quantity of a) k items of list
        further it provides their specific value of quantity
"""
function maxk(a, k)
    b = partialsortperm(a, 1:k, rev=true)
    return b, a[b]
end

function get_x_percent_vulner_single_config(g,flows::Array{Float64,1},loads::Array{Float64,1},x_percent::Float64)
    nr_edges = LightGraphs.ne(g)
    nr_rel_edges = Int(floor(x_percent * nr_edges))
    pos_i, flow_i = maxk(abs.(flows),nr_rel_edges)
    return round(mean(loads[pos_i]), digits=3)
end

"""
interlude: include L measure for paper:
"""
function get_L_for_config(g, P::Array{Float64,1}, C::Array{Float64,1}, hh_battery_capacity::Float64, slack_limit::Float64, slack_idx::Int64, max_momentan_battery_usage::Array{Float64,1}, max_momentan_slack_usage::Array{Float64,1},critical_cases,min_max_mean::String)
    P_origin = copy(P)
    C_origin = copy(C)
    g_origin = copy(g)
    max_momentan_battery_usage_origin = copy(max_momentan_battery_usage)
    max_momentan_slack_usage_origin = copy(max_momentan_slack_usage)
    critical_cases_origin = copy(critical_cases)
    all_edges = collect(LightGraphs.edges(g))
    Ls = Float64[]
    for linefailure_idx in 1:length(all_edges)
        triggered_scr_v = LightGraphs.src(all_edges[linefailure_idx])
        triggered_dst_v = LightGraphs.dst(all_edges[linefailure_idx])
        vertices_initial_failed_edge = [triggered_scr_v, triggered_dst_v]
        connect_components_old = LightGraphs.connected_components(g)
        g, C = linefailure!(g, linefailure_idx, C)
        red_edges = collect(LightGraphs.edges(g))
        g, P, C, max_momentan_battery_usage, max_momentan_slack_usage, critical_cases = accounting_for_possible_strongly_imbalanced_components_slacklimit!(g,P,P_origin,C,hh_battery_capacity,slack_limit,slack_idx,connect_components_old,max_momentan_battery_usage,max_momentan_slack_usage,critical_cases)
        F = flow(g, P)
        if 1 in (abs.(F) .> C)
            connect_components_old = LightGraphs.connected_components(g)
            B = Array(LightGraphs.incidence_matrix(g, oriented=true))
            m = size(B)[2] 
            delete_idxs = Int64[]
            for i in 1:m 
                if abs(F[i]) > C[i] 
                    LightGraphs.rem_edge!(g, findfirst(isodd, B[:, i]), findlast(isodd, B[:, i]))
                    push!(delete_idxs, i)
                end
            end
            deleteat!(C,delete_idxs)
            g, P, C, max_momentan_battery_usage, max_momentan_slack_usage, critical_cases = accounting_for_possible_strongly_imbalanced_components_slacklimit!(g,P,P_origin,C,hh_battery_capacity,slack_limit,slack_idx,connect_components_old,max_momentan_battery_usage,max_momentan_slack_usage,critical_cases)
        end
        all_remaining_edges = collect(LightGraphs.edges(g))
        vertices_of_failed_edges = Int64[]
        for edge in red_edges[red_edges .∉ [all_remaining_edges]]
            start = LightGraphs.src(edge)
            fin = LightGraphs.dst(edge)
            push!(vertices_of_failed_edges, start, fin)
        end
        failure_distances = Int64[]
        if isempty(vertices_of_failed_edges)
            push!(failure_distances, 0)
        else
            for i in 1:length(vertices_initial_failed_edge)
                for j in 1:length(vertices_of_failed_edges)
                    push!(failure_distances, length(a_star(g,vertices_initial_failed_edge[i],vertices_of_failed_edges[j])))
                end
            end
        end
        if min_max_mean == "min"
            push!(Ls, minimum(failure_distances))
        elseif min_max_mean == "max"
            push!(Ls, maximum(failure_distances))
        else 
            push!(Ls, mean(failure_distances))
        end
        P = copy(P_origin)
        C = copy(C_origin)
        g = copy(g_origin)
        max_momentan_battery_usage = copy(max_momentan_battery_usage_origin)
        max_momentan_slack_usage = copy(max_momentan_slack_usage_origin)
        critical_cases = copy(critical_cases_origin)
    end
    Ls_av = mean(Ls)
    Ls, Ls_av
end













include("graph_based_clustering.jl")
include("MCMC_simulation_functions.jl")
include("eval_measures.jl")
include("square_grid_kmeans_clustering.jl")
include("capacity_setting.jl")
include("vertex_locs_and_colour_fillc.jl")
include("new_cascading_algorithm.jl")
include("false_positive_and_negative.jl")
#include("deprecated_unused_functions.jl")

###########################################################################################
#######      generating new topologies with their companion capacity files       ##########
###########################################################################################
##
#C_neu = Capacity_for_specific_PowerGrid()
#C = JLD.load("capacity_for_RandomPowerGrid_N81_for_simulation_test.jld")["C"]
#C = C .+ 2
#C = round.(C; digits = 3)
#JLD.save("capacity_for_RandomPowerGrid_N81_for_simulation_test.jld", "C", C)
#C = JLD.load("capacity_for_RandomPowerGrid_N81_for_simulation_test.jld")["C"]
#C_neu = Capacity_for_square_PowerGrid()
##
""" 
    Creates and saves a real topology graph
    Returns a figure of the graph 
"""
function create_save_plot_realtopology_graph(Filename_graph::String) ##Filename_save needs relative path to e.g. "Analysis_RUNS/Square_Grid/...NAME..."
    fig = Figure(backgroundcolor = RGBf0(0.98, 0.98, 0.98),
        resolution = (1500, 800))

    ax1 = fig[1, 1] = Axis(fig, title = "Embedded Real Power Grid Graph")
    vertex_nr = 81
    ge = gen_power_grid(vertex_nr)
    adj_matrix = LightGraphs.adjacency_matrix(ge)
    ge_neu = LightGraphs.SimpleGraph(adj_matrix)
    embedding = stress(adj_matrix) #spectral(adj_matrix)#spring(adj_matrix)
    lay = (M) -> embedding
    results_mod, number_of_clusters = run_python_modularity(ge)
    nodefillc = nodefillc_modularity(results_mod,vertex_nr)
    save_graph_as_jld(adj_matrix, embedding, results_mod, Filename_graph)
    graph1 = graphplot!(ax1,ge_neu,layout=lay,edge_width=2.0,node_size=12.0,node_color=nodefillc)
    CairoMakie.save(Filename_graph*".pdf", fig)

    adj_matrix_neuer, coordinates_neuer, results_mod_neuer = load_graph_realtopology(Filename_graph)
    if adj_matrix == adj_matrix_neuer && embedding == coordinates_neuer 
        print("Saved Data is correct!")
    end
    fig
end
#create_save_plot_realtopology_graph("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G5/RandomPowerGrid_N81_G5")
#Capacity_for_specific_PowerGrid("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G2/RandomPowerGrid_N81_G2")
#Capacity_for_specific_PowerGrid("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G3/RandomPowerGrid_N81_G3")
#Capacity_for_specific_PowerGrid("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G4/RandomPowerGrid_N81_G4")
#Capacity_for_specific_PowerGrid("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G5/RandomPowerGrid_N81_G5")
#JLD.load("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G3/RandomPowerGrid_N81_G3_capacity.jld")["C"]
"""
    Calls the function 'create_save_plot_realtopology_graph()'
    Calculates an MCMC suitable capacity for the edges of the graph.
    Returns a figure of the graph 
"""
function setting_up_graph_and_capacity_realtopology(Filename_graph::String)
    fig = create_save_plot_realtopology_graph(Filename_graph)
    Capacity_for_specific_PowerGrid(Filename_graph)
    fig
end

""" FOR A SQUARE GRID
    For each set of fixed number of producers and consumers in a grid: (nr-prod, nr-cons); where nr-prod + nr-cons = CONST.
    Returns the average number of edges that connect a consumer to a producer in a grid
"""
function average_nr_cons_prod_edges()
    g = gen_square_grid(9)
    nr_edges = LightGraphs.ne(g)
    nr_vertices = LightGraphs.nv(g)
    result_table = []
    @showprogress for i in collect(1:80)
        nr_prod = i
        nr_cons = 81 - i
        nrs_gen_con = zeros(1000000)
        for j in 1:1000000
            powers = shuffle(vcat(-1 .* ones(nr_prod), ones(nr_cons)))
            nr_gen_gen, nr_con_con, nr_gen_con = nr_gen_con_realtopology(powers, g)
            nrs_gen_con[j] = nr_gen_con
        end
        push!(result_table, round(mean(nrs_gen_con), digits=2))
    end
    return result_table
end
#square_gen_con_edges_average = average_nr_cons_prod_edges()
#JLD2.save("Analysis_RUNS_new/Square_Grid/square_average_nrs_gen_con_edges.jld2", "nrs_gen_con_edges", square_gen_con_edges_average)
#print(square_gen_con_edges_average)

""" FOR A REAL TOPOLOGY GRID
    For each set of fixed number of producers and consumers in a grid: (nr-prod, nr-cons); where nr-prod + nr-cons = CONST.
    Returns the average number of edges that connect a consumer to a producer in a grid
"""
function average_nr_cons_prod_edges_realtoplogy(graph_jld_file::String)
    adj_matrix, coordinates, results_mod = load_graph_realtopology(graph_jld_file)
    g = LightGraphs.SimpleGraph(adj_matrix)
    nr_edges = LightGraphs.ne(g)
    nr_vertices = LightGraphs.nv(g)
    result_table = []
    @showprogress for i in collect(1:80)
        nr_prod = i
        nr_cons = 81 - i
        nrs_gen_con = zeros(1000000)
        for j in 1:1000000
            powers = shuffle(vcat(-1 .* ones(nr_prod), ones(nr_cons)))
            nr_gen_gen, nr_con_con, nr_gen_con = nr_gen_con_realtopology(powers, g)
            nrs_gen_con[j] = nr_gen_con
        end
        push!(result_table, round(mean(nrs_gen_con), digits=2))
    end
    return result_table
end
#G2_gen_con_edges_average = average_nr_cons_prod_edges_realtoplogy("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G2/RandomPowerGrid_N81_G2")
#JLD2.save("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G2/G2_average_nrs_gen_con_edges.jld2", "nrs_gen_con_edges", G2_gen_con_edges_average)
#print(G2_gen_con_edges_average)
#G3_gen_con_edges_average = average_nr_cons_prod_edges_realtoplogy("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G3/RandomPowerGrid_N81_G3")
#JLD2.save("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G3/G3_average_nrs_gen_con_edges.jld2", "nrs_gen_con_edges", G3_gen_con_edges_average)
#print(G3_gen_con_edges_average)
#G4_gen_con_edges_average = average_nr_cons_prod_edges_realtoplogy("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G4/RandomPowerGrid_N81_G4")
#JLD2.save("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G4/G4_average_nrs_gen_con_edges.jld2", "nrs_gen_con_edges", G4_gen_con_edges_average)
#print(G4_gen_con_edges_average)
#G5_gen_con_edges_average = average_nr_cons_prod_edges_realtoplogy("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G5/RandomPowerGrid_N81_G5")
#JLD2.save("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G5/G5_average_nrs_gen_con_edges.jld2", "nrs_gen_con_edges", G5_gen_con_edges_average)
#print(G5_gen_con_edges_average)
#Capacity_for_specific_PowerGrid("Analysis_RUNS/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1")
#setting_up_graph_and_capacity_realtopology("Analysis_RUNS/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1")
##
#testplot("RandomPowerGrid_N81_for_simulation_test")
##
#Filename_input = "Data_Set_OPSD"
#Filename_output = "Analysis_RUNS/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1_simulation_result_test"
#Filename_graph = "Analysis_RUNS/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1"
#time = "10:00:00"
##N_runs = 1
#nr_vertices = 81
#distr_of_node_types = [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2] 
#PVplant_scaling = 3
#init_day = 119
#swap_prob = 1.0
#drift_prob = 0.00033333334 
#vari = 4 
#devi_frac = 0.1
#buffer = 1.0
#annealing_schedule = temp_ex00neu
#k_max = 20
#saving = true
##

"""
    SIMULATION: A seperate MCMC chain is started for each day.
"""
function simulation_days_separate(Filename_input::String, Filename_output::String, Filename_graph::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, hh_battery_capacity::Float64, slack_limit::Float64, annealing_schedule, k_max::Int64)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = LightGraphs.ne(g)#length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
            flush(stdout)
        else 
            print("Capacity works good to start it!")
            flush(stdout)
        end
        println("Len_P:", length(P))
        flush(stdout)
        println("Len_Capacity:", length(C))
        flush(stdout)

        en = Float64[]
        #energy_initial = improved_energy!(g, P, C, hh_battery_capacity)
        energy_initial = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
        g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        P = copy(P_initial)
        push!(en, energy_initial[2])
        
        
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)

        Ps = Array{Float64,1}[]
        push!(Ps, P)
        energies = []
        push!(energies, energy_initial[1])
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flow(g,P),C))

        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)

        @showprogress for l in 0:k_max - 1
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P)
            energy_old = copy(energy_new)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            @label new_config_gen 
            while 1 in (abs.(flow(g, P)) .> C) 
                var_at_place = copy(var_at_place_old)
                dataset = copy(dataset_old)
                drift = copy(drift_old)
                swap = copy(swap_old)
                dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            end
            P_new = copy(P)
            energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            P = copy(P_new)
            ΔE = energy_new[2] - en[end]
            Temp = annealing_schedule(l+1,k_max,drift)
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                        # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #g = LightGraphs.SimpleGraph(adj_matrix)
                    #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                    if 1 in (abs.(flow(g, P)) .> C) ###battery case
                        @goto new_config_gen
                    end
                    P_newtoo = copy(P)
                    energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
                    g = LightGraphs.SimpleGraph(adj_matrix)
                    C = copy(C_old)
                    P = copy(P_newtoo)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            push!(en, energy_new[2])
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            push!(Ps, P)
            push!(energies, energy_new[1])
            push!(line_loadings, eval_line_loadings(flow(g,P),C))
        end
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*".jld2", "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, 
                "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en
                    )
    end
end

function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology(Filename_input::String, Filename_output::String, Filename_graph::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, annealing_schedule, k_max::Int64, saving::Bool)
    Data = []
    lowE_config = []
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    #loading of fixed real topology power grid
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    #cluster_nrs, largest_cluster = size(modularity_result)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
        else 
            print("Capacity works good to start it!")
        end
        println("Len_P:", length(P))
        println("Len_Capacity:", length(C))
        en = Float64[ ]
        clustering_measure_list = Float64[]
        av_red_C = Float64[]
        mean_P = Float64[]
        vari_P = Float64[]
        weighted_pi_sum = Float64[]
        redundant_capacity_value = Float64[]
        #ItterStep_Drift_Enegry = Tuple{Int, Int,Float64}[]
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        N_removals = 0
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l, slack_index)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l, slack_index)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift_realtopology!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l,slack_index)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            #@time begin
            energy_old = energy_realtopology!(g, P_old, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            C = copy(C_old)
            energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - energy_old[2]
            #end
            #### performance: let energy() calculate G_av only
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                     # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x < vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #g = LightGraphs.SimpleGraph(adj_matrix)
                    #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                end
            end
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            clustering_measure = eval_clustering_network_measure_for_P_realtopology(P, C, Filename_graph)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
            #print("slack_pos:", slack_pos)
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,Int(slack_index))
            
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(en, energy_old[2])
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            #### this part extracts the configuration along the way of sampling through the year, that has minimal G_av
            if isempty(lowE_config)
                #print("start")
                data_lowE = P_old, energy_old, nr_gen_con_realtopology(P_old, g), drift_old, dataset_old, av_red_C[end]
                global P_diff_final = P_diff_all_edges_realtopology(P, g)
                global flow_final = flow(g,P)
                global Critical_links_final = eval_critical_links(g,P,C)
                global line_loadings_final = eval_line_loadings(flow_final,C)
                push!(lowE_config, data_lowE)
            else
                if energy_old[2] <= lowE_config[1][2][2]
                    #print("update")
                    data_lowE = P_old, energy_old, nr_gen_con_realtopology(P_old, g), drift_old, dataset_old, av_red_C[end]
                    global P_diff_final = P_diff_all_edges_realtopology(P, g)
                    global flow_final = flow(g,P)
                    global Critical_links_final = eval_critical_links(g,P,C)
                    global line_loadings_final = eval_line_loadings(flow_final,C)
                    lowE_config[1] = data_lowE
                else
                    lowE_config
                end
            end
            #### this part extracts the configuration along the way of sampling through the year, that is not vulnerable 
            #### but has high clustering measure.. if existing --> to check for why it is showing this clustering
            #### while not being vulnerable 
            #if isempty(nonvulnerable_highcluster_config)
            #    #print("start")
            #    data_highGav_highcluster = P_old, energy_old, clustering_measure, drift_old, dataset_old
            #    push!(nonvulnerable_highcluster_config, data_highGav_highcluster)
            #else
            #    if clustering_measure[1] >= nonvulnerable_highcluster_config[1][3][1] && energy_old[2] >= nonvulnerable_highcluster_config[1][2][2]
            #        #print("update")
            #        data_highGav_highcluster = P_old, energy_old, clustering_measure, drift_old, dataset_old
            #        nonvulnerable_highcluster_config[1] = data_highGav_highcluster
            #    else
            #        nonvulnerable_highcluster_config
            #    end
            #end
        end
        g = LightGraphs.SimpleGraph(adj_matrix)
        C = copy(C_old)
        energy_initial = energy_realtopology!(g, P_initial, C, N_removals)
        g = LightGraphs.SimpleGraph(adj_matrix)
        C = copy(C_old)
        energy_final = lowE_config[1][2]
        P_diff_init = P_diff_all_edges_realtopology(P_initial, g)
        flow_init = flow(g,P_initial)
        Critical_links_init = eval_critical_links(g,P_initial,C)
        line_loadings_init = eval_line_loadings(flow(g,P_initial),C)
        g = LightGraphs.SimpleGraph(adj_matrix)
        C = copy(C_old)
        SA_extremal = P_initial, energy_initial, nr_gen_con_realtopology(P_initial, g), P, energy_final, nr_gen_con_realtopology(P, g), Drift_evol, Swap_evol, en, flow_init, flow_final, P_diff_init, P_diff_final, Critical_links_init, Critical_links_final, lowE_config, mean_P, vari_P, weighted_pi_sum, clustering_measure_list, redundant_capacity_value, line_loadings_init, line_loadings_final, Counter, av_red_C
        print("Capacity Violation in each MC Step:", Counter)
        push!(Data, SA_extremal)
    end 
    if saving == true
        JLD.save(Filename_output*".jld", "Data", Data, "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types, "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob, "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer)
    end
    Data
end

""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_new(Filename_input::String, Filename_output::String, Filename_graph::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, annealing_schedule, k_max::Int64, saving::Bool)
    Data = []
    lowE_config = []
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    #loading of fixed real topology power grid
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    #cluster_nrs, largest_cluster = size(modularity_result)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = LightGraphs.ne(g)#length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
        else 
            print("Capacity works good to start it!")
        end
        println("Len_P:", length(P))
        println("Len_Capacity:", length(C))
        en = Float64[]
        clustering_measure_list = Float64[]
        av_red_C = Float64[]
        mean_P = Float64[]
        vari_P = Float64[]
        weighted_pi_sum = Float64[]
        redundant_capacity_value = Float64[]
        #ItterStep_Drift_Enegry = Tuple{Int, Int,Float64}[]
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        clustering_measure = eval_clustering_network_measure_for_P_realtopology(P_initial, C, Filename_graph)
        push!(clustering_measure_list, clustering_measure)
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,Int(slack_index))
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        energy_initial = energy_realtopology!(g, P_initial, C, N_removals)
        g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        push!(en, energy_initial[2])
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        P_diff_init = P_diff_all_edges_realtopology(P_initial, g)
        flow_init = flow(g,P_initial)
        Critical_links_init = eval_critical_links(g,P_initial,C)
        line_loadings_init = eval_line_loadings(flow_init,C)
        g = LightGraphs.SimpleGraph(adj_matrix) # kann evtl weg
        C = copy(C_old) #kann evtl weg
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)
        
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift_realtopology!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1,slack_index)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            energy_old = copy(energy_new)
            #@time begin
            #energy_old = en[end]#energy_realtopology!(g, P_old, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - en[end]
            #end
            #### performance: let energy() calculate G_av only
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                     # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #g = LightGraphs.SimpleGraph(adj_matrix)
                    #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                    energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
                    g = LightGraphs.SimpleGraph(adj_matrix)
                    C = copy(C_old)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            clustering_measure = eval_clustering_network_measure_for_P_realtopology(P, C, Filename_graph)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            #print("slack_pos:", slack_pos)
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,Int(slack_index))
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(en, energy_new[2])
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            #### this part extracts the configuration along the way of sampling through the year, that has minimal G_av
            if isempty(lowE_config)
                #print("start")
                data_lowE = P, energy_new, nr_gen_con_realtopology(P, g), drift, dataset, av_red_C[end]
                global P_diff_final = P_diff_all_edges_realtopology(P, g)
                global flow_final = flow(g,P)
                global Critical_links_final = eval_critical_links(g,P,C)
                global line_loadings_final = eval_line_loadings(flow_final,C)
                push!(lowE_config, data_lowE)
            else
                if energy_new[2] <= lowE_config[1][2][2]
                    #print("update")
                    data_lowE = P, energy_new, nr_gen_con_realtopology(P, g), drift, dataset, av_red_C[end]
                    global P_diff_final = P_diff_all_edges_realtopology(P, g)
                    global flow_final = flow(g,P)
                    global Critical_links_final = eval_critical_links(g,P,C)
                    global line_loadings_final = eval_line_loadings(flow_final,C)
                    lowE_config[1] = data_lowE
                else
                    lowE_config
                end
            end
            #### this part extracts the configuration along the way of sampling through the year, that is not vulnerable 
            #### but has high clustering measure.. if existing --> to check for why it is showing this clustering
            #### while not being vulnerable 
            #if isempty(nonvulnerable_highcluster_config)
            #    #print("start")
            #    data_highGav_highcluster = P_old, energy_old, clustering_measure, drift_old, dataset_old
            #    push!(nonvulnerable_highcluster_config, data_highGav_highcluster)
            #else
            #    if clustering_measure[1] >= nonvulnerable_highcluster_config[1][3][1] && energy_old[2] >= nonvulnerable_highcluster_config[1][2][2]
            #        #print("update")
            #        data_highGav_highcluster = P_old, energy_old, clustering_measure, drift_old, dataset_old
            #        nonvulnerable_highcluster_config[1] = data_highGav_highcluster
            #    else
            #        nonvulnerable_highcluster_config
            #    end
            #end
        end
        energy_final = lowE_config[1][2]
        g = LightGraphs.SimpleGraph(adj_matrix) #evtl weg
        C = copy(C_old) #evtl weg
        SA_extremal = P_initial, energy_initial, nr_gen_con_realtopology(P_initial, g), P, energy_final, nr_gen_con_realtopology(P, g), Drift_evol, Swap_evol, en, flow_init, flow_final, P_diff_init, P_diff_final, Critical_links_init, Critical_links_final, lowE_config, mean_P, vari_P, weighted_pi_sum, clustering_measure_list, redundant_capacity_value, line_loadings_init, line_loadings_final, Counter, av_red_C
        print("Capacity Violation in each MC Step:", Counter)
        push!(Data, SA_extremal)
    end 
    if saving == true
        JLD.save(Filename_output*".jld", "Data", Data, "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types, "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob, "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer)
    end
    Data
end
#print("hello")
#combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_new("Data_Set_OPSD", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1_simulation_result_test", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1", "10:00:00", 1, 81, [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2], 3, 119, 1.0, 0.0003333333, 4, 0.1, 1.0, temp_ex00neu, 12, true)
""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_newcascading(Filename_input::String, Filename_output::String, Filename_graph::String, Filename_gen_con::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, hh_battery_capacity::Float64, slack_limit::Float64, annealing_schedule, k_max::Int64, saving::Bool)
    Data = []
    lowE_config = []
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    #loading of fixed real topology power grid
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = LightGraphs.ne(g)#length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
            flush(stdout)
        else 
            print("Capacity works good to start it!")
            flush(stdout)
        end
        println("Len_P:", length(P))
        flush(stdout)
        println("Len_Capacity:", length(C))
        flush(stdout)

        en = Float64[]
        energy_initial = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
        g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        P = copy(P_initial)
        push!(en, energy_initial[2])

        #relevant networkmeasure components 
        vari_P = Float64[]
        P_copy = copy(P_initial)
        P_var = round(var(deleteat!(P_copy,Int(slack_index))), digits=3)
        push!(vari_P, P_var)

        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        cluster_measure = Float64[]
        nr_prod = length(findall(P .> 0))
        av_nr_gen_con = G1_gen_con_edges_average[nr_prod]
        gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)
        nrs_gen_con = gen_con
        push!(cluster_measure, round(av_nr_gen_con / nrs_gen_con, digits=3))

        mean_line_loadings = Float64[]
        line_loads = eval_line_loadings(flow(g,P),C)
        mean_line_loads = round(mean(line_loads), digits=3)
        push!(mean_line_loadings, mean_line_loads)
        
        
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)
        
        @showprogress for l in 0:k_max - 1
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            energy_old = copy(energy_new)
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            @label new_config_gen ###battery case
            while 1 in (abs.(flow(g, P)) .> C) ###battery case
                var_at_place = copy(var_at_place_old)
                dataset = copy(dataset_old)
                drift = copy(drift_old)
                swap = copy(swap_old)
                #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
                dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            end
            P_new = copy(P)
            energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            P = copy(P_new)
            ΔE = energy_new[2] - en[end]
            #Temp = annealing_schedule(l+1,k_max,drift)
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                     # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                    if 1 in (abs.(flow(g, P)) .> C) ###battery case
                        @goto new_config_gen
                    end
                    P_newtoo = copy(P)
                    energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
                    g = LightGraphs.SimpleGraph(adj_matrix)
                    C = copy(C_old)
                    P = copy(P_newtoo)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            push!(en, energy_new[2])
            
            P_copy = copy(P)
            P_var = round(var(deleteat!(P_copy,Int(slack_index))), digits=3)
            push!(vari_P, P_var)

            nr_prod = length(findall(P .> 0))
            av_nr_gen_con = G1_gen_con_edges_average[nr_prod]
            gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)
            nrs_gen_con = gen_con
            push!(cluster_measure, round(av_nr_gen_con / nrs_gen_con, digits=3))

            line_loads = eval_line_loadings(flow(g,P),C)
            mean_line_loads = round(mean(line_loads), digits=3)
            push!(mean_line_loadings, mean_line_loads)

            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            
            #### this part extracts the configuration along the way of sampling through the year, that has minimal G_av
            if isempty(lowE_config)
                data_lowE = P, energy_new, l, drift, P_var, line_loads, cluster_measure[end]
                push!(lowE_config, data_lowE)
            else
                if energy_new[2] <= lowE_config[1][2][2]
                    data_lowE = P, energy_new, l, drift, P_var, line_loads, cluster_measure[end]
                    lowE_config[1] = data_lowE
                else
                    lowE_config
                end
            end
        end
        SA_extremal = P_initial, energy_initial, lowE_config, Drift_evol, Swap_evol, en, vari_P, cluster_measure, mean_line_loadings
        push!(Data, SA_extremal)
    end 
    if saving == true
        JLD2.save(Filename_output*".jld2", "Data", Data, "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types, "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob, "variance", vari, "deviation_fraction", devi_frac, "hh_battery_capacity", hh_battery_capacity, "slack_limit", slack_limit)
    end
end


#Filename_input = "Data_Set_OPSD"
#Filename_output = "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_RUN4_nce_119_hh_d500_s1_sall_test"
#Filename_graph = "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1"
#time = "10:00:00"
#N_runs = 1
#nr_vertices = 81
#distr_of_node_types = [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2]
#PVplant_scaling = 3
#init_day = 119
#swap_prob =  1.0
#drift_prob = 0.002
#vari = 4
#devi_frac = 0.1 
#buffer = 0.92
#annealing_schedule = temp_ex00
#k_max = 3 
#saving = true

""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_new_saveall(Filename_input::String, Filename_output::String, Filename_graph::String, Filename_gen_con::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, annealing_schedule, k_max::Int64, saving::Bool)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    #loading of fixed real topology power grid
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    #cluster_nrs, largest_cluster = size(modularity_result)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = LightGraphs.ne(g)#length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
            flush(stdout)
        else 
            print("Capacity works good to start it!")
            flush(stdout)
        end
        println("Len_P:", length(P))
        flush(stdout)
        println("Len_Capacity:", length(C))
        flush(stdout)

        en = Float64[]
        energy_initial = energy_with_battery_check!(g,P,C) ###battery case
        #energy_initial = energy_realtopology!(g, P_initial, C, N_removals)
        g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        push!(en, energy_initial[2])
        clustering_measure_list = Float64[]
        clustering_measure = eval_clustering_network_measure_for_P_realtopology(P_initial, C, Filename_graph)
        push!(clustering_measure_list, clustering_measure)
        mean_P = Float64[]
        vari_P = Float64[]
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,Int(slack_index))
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        weighted_pi_sum = Float64[]
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        redundant_capacity_value = Float64[]
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        av_red_C = Float64[]
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        Ps = Array{Float64,1}[]
        push!(Ps, P)
        energies = []
        push!(energies, energy_initial[1])
        flows = Array{Float64,1}[]
        push!(flows, flow(g,P))
        P_diffs = Array{Float64,1}[]
        push!(P_diffs, P_diff_all_edges_realtopology(P, g))
        Critical_links = Array{Any,1}[]
        non_reroutable_flow = Array{Any,1}[]
        critical_links_result = eval_critical_links(g,P,C)
        push!(Critical_links, critical_links_result[1])
        push!(non_reroutable_flow, critical_links_result[2])
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flows[end],C))
        max_battery_usages = Array{Float64,1}[]  ###battery case
        push!(max_battery_usages, energy_initial[3]) ###battery case
        av_max_battery_usage = Float64[] ###battery case
        push!(av_max_battery_usage, energy_initial[4]) ###battery case
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            energy_old = copy(energy_new)
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            @label new_config_gen ###battery case
            while 1 in (abs.(flow(g, P)) .> C) ###battery case
                var_at_place = copy(var_at_place_old)
                dataset = copy(dataset_old)
                drift = copy(drift_old)
                swap = copy(swap_old)
                dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            end
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift_realtopology!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1,slack_index)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            #@time begin
            #energy_old = en[end]#energy_realtopology!(g, P_old, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            energy_new = energy_with_battery_check!(g,P,C) ###battery case
            #energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - en[end]
            #end
            #### performance: let energy() calculate G_av only
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                        # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #g = LightGraphs.SimpleGraph(adj_matrix)
                    #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                    if 1 in (abs.(flow(g, P)) .> C) ###battery case
                        @goto new_config_gen
                    end
                    energy_new = energy_with_battery_check!(g,P,C) ###battery case
                    #energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
                    g = LightGraphs.SimpleGraph(adj_matrix)
                    C = copy(C_old)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            push!(en, energy_new[2])
            clustering_measure = eval_clustering_network_measure_for_P_realtopology(P, C, Filename_graph)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            #print("slack_pos:", slack_pos)
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,Int(slack_index))
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(Ps, P)
            push!(energies, energy_new[1])
            push!(flows, flow(g,P))
            push!(P_diffs, P_diff_all_edges_realtopology(P, g))
            critical_links_result = eval_critical_links(g,P,C)
            push!(Critical_links, critical_links_result[1])
            push!(non_reroutable_flow, critical_links_result[2])
            push!(line_loadings, eval_line_loadings(flows[end],C))
            push!(max_battery_usages,energy_new[3]) ###battery case
            push!(av_max_battery_usage, energy_new[4]) ###battery case
        end
        ######### EVAL MEASURES SECTION ##########
        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        av_nr_gen_con = zeros(length(Ps))
        nrs_gen_con = zeros(length(Ps))
        for (i,P) in enumerate(Ps)
            gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)
            nrs_gen_con[i] = gen_con
            nr_prod = length(findall(P .> 0))
            av_nr_gen_con[i] = G1_gen_con_edges_average[nr_prod]
        end
        stable_pos = findall(x->x==maximum(en), en)
        mean_line_loadings = round.(mean.(line_loadings), digits=3)
        max_lineload_at_stable = maximum(mean_line_loadings[stable_pos])
        max_pmean_at_stable = maximum(mean_P[stable_pos])
        measures, measures_string = define_vulnerability_measures(vari_P,mean_P,max_pmean_at_stable,clustering_measure_list,
                                    redundant_capacity_value,av_red_C,av_nr_gen_con,
                                    nrs_gen_con,mean_line_loadings,max_lineload_at_stable,weighted_pi_sum)
        FP_Data = eval_FP_for_measures(en, measures, energies)

        JLD2.save(Filename_output*"_FP_Analysis.jld2", "FP_Analysis", FP_Data, "nrs_gen_con", nrs_gen_con, "av_nr_gen_con", av_nr_gen_con, 
                    "mean_line_loadings", mean_line_loadings, "max_lineload_at_stable", max_lineload_at_stable, "max_pmean_at_stable", max_pmean_at_stable,
                    "measures", measures, "measures_string", measures_string)
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_flows.jld2", "flows", flows)
        JLD2.save(Filename_output*"_Pdiffs.jld2", "P_diffs", P_diffs)
        JLD2.save(Filename_output*"_Critical_links.jld2", "Critical_links", Critical_links)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*"_battery.jld2", "max_battery_usages", max_battery_usages, "av_max_battery_usage", av_max_battery_usage) ###battery case
        JLD2.save(Filename_output*".jld2", "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, 
                "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en, "mean_P", mean_P, "vari_P", vari_P, "weighted_pi_sum", weighted_pi_sum, "clustering_measure_list", clustering_measure_list, 
                    "redundant_capacity_value", redundant_capacity_value, "Counter", Counter, "av_red_C", av_red_C, "non_reroutable_flow", non_reroutable_flow
                    )
        print("Capacity Violation in each MC Step:", Counter)
        print("not able to redistribute flow for lines that are overloaded from the beginning:", non_reroutable_flow)
    end 
end
#print("hello")
#combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_new_saveall("Data_Set_OPSD", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_RUN4_nce_119_hh_d500_s1_sall_test", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_average_nrs_gen_con_edges", "10:00:00", 1, 81, [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2], 3, 119, 1.0, 0.002, 4, 0.1, 1.0, temp_ex00, 13, true)

""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_newcascading_saveall(Filename_input::String, Filename_output::String, Filename_graph::String, Filename_gen_con::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, hh_battery_capacity::Float64, slack_limit::Float64, annealing_schedule, k_max::Int64, saving::Bool)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    #loading of fixed real topology power grid
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    #cluster_nrs, largest_cluster = size(modularity_result)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = LightGraphs.ne(g)#length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
            flush(stdout)
        else 
            print("Capacity works good to start it!")
            flush(stdout)
        end
        println("Len_P:", length(P))
        flush(stdout)
        println("Len_Capacity:", length(C))
        flush(stdout)

        en = Float64[]
        #energy_initial = improved_energy!(g, P, C, hh_battery_capacity)
        energy_initial = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
        g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        P = copy(P_initial)
        push!(en, energy_initial[2])
        clustering_measure_list = Float64[]
        clustering_measure = eval_clustering_network_measure_for_P_realtopology(P_initial, C, Filename_graph)
        push!(clustering_measure_list, clustering_measure)
        mean_P = Float64[]
        vari_P = Float64[]
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,Int(slack_index))
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        weighted_pi_sum = Float64[]
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        redundant_capacity_value = Float64[]
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        av_red_C = Float64[]
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        Ps = Array{Float64,1}[]
        push!(Ps, P)
        energies = []
        push!(energies, energy_initial[1])
        flows = Array{Float64,1}[]
        push!(flows, flow(g,P))
        P_diffs = Array{Float64,1}[]
        push!(P_diffs, P_diff_all_edges_realtopology(P, g))
        Critical_links = Array{Any,1}[]
        non_reroutable_flow = Array{Any,1}[]
        critical_links_result = eval_critical_links(g,P,C)
        push!(Critical_links, critical_links_result[1])
        push!(non_reroutable_flow, critical_links_result[2])
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flows[end],C))
        max_battery_usages = Float64[]  ###battery case
        push!(max_battery_usages, energy_initial[3]) ###battery case
        av_max_battery_usage = Float64[] ###battery case
        push!(av_max_battery_usage, energy_initial[4]) ###battery case
        max_slack_usages = Float64[]  ###battery case
        push!(max_slack_usages, energy_initial[5]) ###battery case
        av_max_slack_usage = Float64[] ###battery case
        push!(av_max_slack_usage, energy_initial[6]) ###battery case
        critical_cases = Vector{Tuple{Float64,Int64}}[] ###battery case
        push!(critical_cases, energy_initial[7]) ###battery case
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)

        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            energy_old = copy(energy_new)
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            @label new_config_gen ###battery case
            while 1 in (abs.(flow(g, P)) .> C) ###battery case
                var_at_place = copy(var_at_place_old)
                dataset = copy(dataset_old)
                drift = copy(drift_old)
                swap = copy(swap_old)
                #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
                dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            end
            P_new = copy(P)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift_realtopology!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1,slack_index)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            #energy_old = en[end]#energy_realtopology!(g, P_old, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            #energy_new = improved_energy!(g, P, C, hh_battery_capacity) #[2] # by [2] only the second value of tuple is returned (G_av)
            energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            P = copy(P_new)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - en[end]
            #end
            #### performance: let energy() calculate G_av only
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                        # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #g = LightGraphs.SimpleGraph(adj_matrix)
                    #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                    if 1 in (abs.(flow(g, P)) .> C) ###battery case
                        @goto new_config_gen
                    end
                    P_newtoo = copy(P)
                    #energy_new = improved_energy!(g, P, C, hh_battery_capacity) #[2] # by [2] only the second value of tuple is returned (G_av)
                    energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
                    g = LightGraphs.SimpleGraph(adj_matrix)
                    C = copy(C_old)
                    P = copy(P_newtoo)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            push!(en, energy_new[2])
            clustering_measure = eval_clustering_network_measure_for_P_realtopology(P, C, Filename_graph)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            #print("slack_pos:", slack_pos)
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,Int(slack_index))
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(Ps, P)
            push!(energies, energy_new[1])
            push!(flows, flow(g,P))
            push!(P_diffs, P_diff_all_edges_realtopology(P, g))
            critical_links_result = eval_critical_links(g,P,C)
            push!(Critical_links, critical_links_result[1])
            push!(non_reroutable_flow, critical_links_result[2])
            push!(line_loadings, eval_line_loadings(flows[end],C))
            push!(max_battery_usages, energy_new[3])
            push!(av_max_battery_usage, energy_new[4])
            push!(max_slack_usages, energy_new[5])
            push!(av_max_slack_usage, energy_new[6])
            push!(critical_cases, energy_new[7])
        end
        ######### EVAL MEASURES SECTION ##########
        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        av_nr_gen_con = zeros(length(Ps))
        nrs_gen_con = zeros(length(Ps))
        for (i,P) in enumerate(Ps)
            gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)
            nrs_gen_con[i] = gen_con
            nr_prod = length(findall(P .> 0))
            av_nr_gen_con[i] = G1_gen_con_edges_average[nr_prod]
        end
        stable_pos = findall(x->x==maximum(en), en)
        mean_line_loadings = round.(mean.(line_loadings), digits=3)
        max_lineload_at_stable = maximum(mean_line_loadings[stable_pos])
        max_pmean_at_stable = maximum(mean_P[stable_pos])
        measures, measures_string = define_vulnerability_measures(vari_P,mean_P,max_pmean_at_stable,clustering_measure_list,
                                    redundant_capacity_value,av_red_C,av_nr_gen_con,
                                    nrs_gen_con,mean_line_loadings,max_lineload_at_stable,weighted_pi_sum)
        FP_Data = eval_FP_for_measures(en, measures, energies)
        JLD2.save(Filename_output*"_FP_Analysis.jld2", "FP_Analysis", FP_Data, "nrs_gen_con", nrs_gen_con, "av_nr_gen_con", av_nr_gen_con, 
                    "mean_line_loadings", mean_line_loadings, "max_lineload_at_stable", max_lineload_at_stable, "max_pmean_at_stable", max_pmean_at_stable,
                    "measures", measures, "measures_string", measures_string)
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_flows.jld2", "flows", flows)
        JLD2.save(Filename_output*"_Pdiffs.jld2", "P_diffs", P_diffs)
        JLD2.save(Filename_output*"_Critical_links.jld2", "Critical_links", Critical_links)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*"_battery_slack.jld2", "max_battery_usages", max_battery_usages, "av_max_battery_usage", av_max_battery_usage,
                         "max_slack_usages", max_slack_usages, "av_max_slack_usage", av_max_slack_usage, "critical_cases", critical_cases)
        JLD2.save(Filename_output*".jld2", "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, 
                "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en, "mean_P", mean_P, "vari_P", vari_P, "weighted_pi_sum", weighted_pi_sum, "clustering_measure_list", clustering_measure_list, 
                    "redundant_capacity_value", redundant_capacity_value, "Counter", Counter, "av_red_C", av_red_C, "non_reroutable_flow", non_reroutable_flow
                    )
        print("Capacity Violation in each MC Step:", Counter)
        print("not able to redistribute flow for lines that are overloaded from the beginning:", non_reroutable_flow)
    end
end
#combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_newcascading_saveall("Data_Set_OPSD", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_RUN15_119_hh_d500_s1_sall_test_forwards", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_average_nrs_gen_con_edges", "10:00:00", 1, 81, [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2], 3, 119, 1.0, 0.002, 4, 0.1, 1.0, 2.151, 205.0, temp_ex00, 99, false)
Base.copy(t::Tuple) = t
#print("hello")
#combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_new_saveall("Data_Set_OPSD", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_RUN4_nce_119_hh_d500_s1_sall_test", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1", "10:00:00", 1, 81, [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2], 3, 119, 1.0, 0.002, 4, 0.1, 1.0, temp_ex00, 3, true)
""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_newcascading_inverse_saveall(Filename_input::String, Filename_output::String, Filename_graph::String, Filename_gen_con::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, hh_battery_capacity::Float64, slack_limit::Float64, annealing_schedule, k_max::Int64, saving::Bool)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = LightGraphs.ne(g)#length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
            flush(stdout)
        else 
            print("Capacity works good to start it!")
            flush(stdout)
        end
        println("Len_P:", length(P))
        flush(stdout)
        println("Len_Capacity:", length(C))
        flush(stdout)

        en = Float64[]
        #energy_initial = improved_energy!(g, P, C, hh_battery_capacity)
        energy_initial = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
        g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        P = copy(P_initial)
        push!(en, energy_initial[2])
        
        P_copy = copy(P_initial)
        P_var = var(deleteat!(P_copy,Int(slack_index)))
        
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        
        Ps = Array{Float64,1}[]
        push!(Ps, P)

        energies = []
        push!(energies, energy_initial[1])

        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        nr_prod = length(findall(P .> 0))
        av_nr_gen_con = G1_gen_con_edges_average[nr_prod]
        gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)

        flows = Array{Float64,1}[]
        push!(flows, flow(g,P))
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flows[end],C))
        mean_line_loadings = mean(line_loadings[end])
        max_lineload_at_stable = 0.234

        Critical_links = Array{Any,1}[]
        non_reroutable_flow = Array{Any,1}[]
        critical_links_result = eval_critical_links(g,P,C)
        push!(Critical_links, critical_links_result[1])
        push!(non_reroutable_flow, critical_links_result[2])
        
        comb_measure_list = Float64[]
        comb_measure = round(P_var * (av_nr_gen_con / gen_con) * (mean_line_loadings / max_lineload_at_stable), digits=3)
        push!(comb_measure_list, comb_measure)

        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        comb_measure_new = copy(comb_measure_list[end])

        @showprogress for l in 0:k_max - 1
            #Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            comb_measure_old = copy(comb_measure_new)
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            @label new_config_gen ###battery case
            while 1 in (abs.(flow(g, P)) .> C) ###battery case
                var_at_place = copy(var_at_place_old)
                dataset = copy(dataset_old)
                drift = copy(drift_old)
                swap = copy(swap_old)
                #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
                dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            end
            P_copy = copy(P)
            P_var = var(deleteat!(P_copy,Int(slack_index)))
            nr_prod = length(findall(P .> 0))
            av_nr_gen_con = G1_gen_con_edges_average[nr_prod]
            gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)
            mean_line_loadings = mean(eval_line_loadings(flow(g,P),C))
            max_lineload_at_stable = 0.234
            comb_measure_new = round(P_var * (av_nr_gen_con / gen_con) * (mean_line_loadings / max_lineload_at_stable), digits=3)
            ΔE = (comb_measure_list[end] - comb_measure_new) * 10.0
            Temp = annealing_schedule(l+1,k_max,drift)
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                        # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                comb_measure_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                comb_measure_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #g = LightGraphs.SimpleGraph(adj_matrix)
                    #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                    if 1 in (abs.(flow(g, P)) .> C) ###battery case
                        @goto new_config_gen
                    end
                    P_copy = copy(P)
                    P_var = var(deleteat!(P_copy,Int(slack_index)))
                    nr_prod = length(findall(P .> 0))
                    av_nr_gen_con = G1_gen_con_edges_average[nr_prod]
                    gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)
                    mean_line_loadings = mean(eval_line_loadings(flow(g,P),C))
                    max_lineload_at_stable = 0.234
                    comb_measure_new = round(P_var * (av_nr_gen_con / gen_con) * (mean_line_loadings / max_lineload_at_stable), digits=3)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    comb_measure_new = copy(comb_measure_old)
                end
            end
            P_new = copy(P)
            energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            P = copy(P_new)
            push!(en, energy_new[2])
            
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            
            push!(Ps, P)
            push!(energies, energy_new[1])

            push!(flows, flow(g,P))
            push!(line_loadings, eval_line_loadings(flows[end],C))
            
            critical_links_result = eval_critical_links(g,P,C)
            push!(Critical_links, critical_links_result[1])
            push!(non_reroutable_flow, critical_links_result[2])
            
            push!(comb_measure_list, comb_measure_new)
        end
        JLD2.save(Filename_output*"_COMB_Measure.jld2", "comb_measure", comb_measure_list)
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_flows.jld2", "flows", flows)
        JLD2.save(Filename_output*"_Critical_links.jld2", "Critical_links", Critical_links)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*".jld2", "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, 
                "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en, "non_reroutable_flow", non_reroutable_flow
                    )
        print("not able to redistribute flow for lines that are overloaded from the beginning:", non_reroutable_flow)
    end
end
#combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_newcascading_inverse_saveall("Data_Set_OPSD", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_RUN15_119_hh_d500_s1_sall_test_inverse", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_average_nrs_gen_con_edges", "10:00:00", 1, 81, [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2], 3, 119, 1.0, 0.002, 4, 0.1, 1.0, 2.151, 205.0, temp_ex00_steps, 99, true)

""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_new_saveall_critlink(Filename_input::String, Filename_output::String, Filename_graph::String, Filename_gen_con::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, annealing_schedule, k_max::Int64, saving::Bool)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    #loading of fixed real topology power grid
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    #cluster_nrs, largest_cluster = size(modularity_result)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = LightGraphs.ne(g)#length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
            flush(stdout)
        else 
            print("Capacity works good to start it!")
            flush(stdout)
        end
        println("Len_P:", length(P))
        flush(stdout)
        println("Len_Capacity:", length(C))
        flush(stdout)

        en = Float64[]
        energy_initial = energy_realtopology!(g, P_initial, C, N_removals)
        g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        push!(en, energy_initial[2])
        clustering_measure_list = Float64[]
        clustering_measure = eval_clustering_network_measure_for_P_realtopology(P_initial, C, Filename_graph)
        push!(clustering_measure_list, clustering_measure)
        mean_P = Float64[]
        vari_P = Float64[]
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,Int(slack_index))
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        weighted_pi_sum = Float64[]
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        redundant_capacity_value = Float64[]
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        av_red_C = Float64[]
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        Ps = Array{Float64,1}[]
        push!(Ps, P)
        energies = []
        push!(energies, energy_initial[1])
        flows = Array{Float64,1}[]
        push!(flows, flow(g,P))
        P_diffs = Array{Float64,1}[]
        push!(P_diffs, P_diff_all_edges_realtopology(P, g))
        Critical_links = Array{Any,1}[]
        critlink_orig = Float64[]
        non_reroutable_flow = Array{Any,1}[]
        critical_links_result = eval_critical_links(g,P,C)
        push!(Critical_links, critical_links_result[1])
        push!(critlink_orig, round(mean(critical_links_result[1][findall(x->x!="inf", critical_links_result[1])]), digits=3))
        push!(non_reroutable_flow, critical_links_result[2])
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flows[end],C))
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        critical_links_result_new = copy(critical_links_result)
        critlink_new = copy(critlink_orig[end])
        
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift_realtopology!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1,slack_index)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            critical_links_result_old = copy(critical_links_result_new)
            #@time begin
            #energy_old = en[end]#energy_realtopology!(g, P_old, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            critical_links_result_new = eval_critical_links(g,P,C)
            critlink_new = round(mean(critical_links_result_new[1][findall(x->x!="inf", critical_links_result_new[1])]), digits=3)
            ##print("I am done with energy_new")
            ΔE = (critlink_orig[end] - critlink_new) * 100.0
            #end
            #### performance: let energy() calculate G_av only
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                        # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                critical_links_result_new
                critlink_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                critical_links_result_new
                critlink_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #g = LightGraphs.SimpleGraph(adj_matrix)
                    #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                    critical_links_result_new = eval_critical_links(g,P,C)
                    critlink_new = round(mean(critical_links_result_new[1][findall(x->x!="inf", critical_links_result_new[1])]), digits=3)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    critical_links_result_new = copy(critical_links_result_old)
                    critlink_new = critlink_orig[end]
                end
            end
            energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            push!(en, energy_new[2])
            clustering_measure = eval_clustering_network_measure_for_P_realtopology(P, C, Filename_graph)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            #print("slack_pos:", slack_pos)
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,Int(slack_index))
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(Ps, P)
            push!(energies, energy_new[1])
            push!(flows, flow(g,P))
            push!(P_diffs, P_diff_all_edges_realtopology(P, g))
            push!(critlink_orig, round(mean(critical_links_result_new[1][findall(x->x!="inf", critical_links_result_new[1])]), digits=3))
            push!(Critical_links, critical_links_result_new[1])
            push!(non_reroutable_flow, critical_links_result_new[2])
            push!(line_loadings, eval_line_loadings(flows[end],C))
        end
        ######### EVAL MEASURES SECTION ##########
        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        av_nr_gen_con = zeros(length(Ps))
        nrs_gen_con = zeros(length(Ps))
        for (i,P) in enumerate(Ps)
            gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)
            nrs_gen_con[i] = gen_con
            nr_prod = length(findall(P .> 0))
            av_nr_gen_con[i] = G1_gen_con_edges_average[nr_prod]
        end
        above_10_cases = zeros(length(critlink_orig))
        critlink = zeros(length(critlink_orig))
        for (i,elem) in enumerate(Critical_links)
            non_inf = elem[findall(x->x!="inf", elem)]
            extreme_case_nr = count(non_inf .> 10.0)
            above_10_cases[i] = extreme_case_nr
            critlink[i] = round(mean(vcat(non_inf[non_inf .<= 10.0], fill(10.0,extreme_case_nr))), digits=3) 
        end
        x_thr = get_X_threshold_critlink(Critical_links, critlink, 1.0)
        stable_pos = findall(x->x <= x_thr, critlink)
        mean_line_loadings = round.(mean.(line_loadings), digits=3)
        max_lineload_at_stable = maximum(mean_line_loadings[stable_pos])
        max_pmean_at_stable = maximum(mean_P[stable_pos])
        measures, measures_string = define_vulnerability_measures_critlink(vari_P,mean_P,max_pmean_at_stable,clustering_measure_list,
                                    redundant_capacity_value,av_red_C,av_nr_gen_con,
                                    nrs_gen_con,mean_line_loadings,max_lineload_at_stable,weighted_pi_sum)
        FP_Data = eval_FP_for_measures_critlink(critlink, measures, critical_links)

        JLD2.save(Filename_output*"_FP_Analysis.jld2", "FP_Analysis", FP_Data, "nrs_gen_con", nrs_gen_con, "av_nr_gen_con", av_nr_gen_con, 
                    "mean_line_loadings", mean_line_loadings, "max_lineload_at_stable", max_lineload_at_stable, "max_pmean_at_stable", max_pmean_at_stable, 
                    "measures", measures, "measures_string", measures_string)
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_flows.jld2", "flows", flows)
        JLD2.save(Filename_output*"_Pdiffs.jld2", "P_diffs", P_diffs)
        JLD2.save(Filename_output*"_Critical_links.jld2", "Critical_links", Critical_links)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*".jld2", "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, "critlink", critlink_orig,
                "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en, "mean_P", mean_P, "vari_P", vari_P, "weighted_pi_sum", weighted_pi_sum, "clustering_measure_list", clustering_measure_list, 
                    "redundant_capacity_value", redundant_capacity_value, "Counter", Counter, "av_red_C", av_red_C, "non_reroutable_flow", non_reroutable_flow
                    )
        print("Capacity Violation in each MC Step:", Counter)
        print("not able to redistribute flow for lines that are overloaded from the beginning:", non_reroutable_flow)
    end 
end
#combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_new_saveall_critlink("Data_Set_OPSD", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_RUN7_nce_119_hh_d500_s1_CriticLink_energy_test", "Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/RandomPowerGrid_N81_G1","Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G2/G2_average_nrs_gen_con_edges", "10:00:00", 1, 81, [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2], 3, 119, 1.0, 0.0002, 4, 0.1, 1.0, temp_ex00, 99, true)
#Crit_linx = JLD2.load("Analysis_RUNS_new/Real_Topology_Grid/RandomPowerGrid_N81_G1/G1_RUN7_nce_119_hh_d500_s1_CriticLink_energy_test_Critical_links.jld2")["Critical_links"]
##
#Filename_C = "Analysis_RUNS/Square_Grid/capacity_square_grid"
#N_side = 9
#T = 1.0
##
""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift(Filename_input::String, Filename_output::String, Filename_C::String, time::String, N_runs::Int64, N_side::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, T::Float64, annealing_schedule, k_max::Int64, saving::Bool)
    Data = []
    lowE_config = []
    #nonvulnerable_highcluster_config = []
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    sample_amount = N_side * N_side
    dataset = init_injection(Filename_input, distr_of_node_types, sample_amount, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    P = initial_inject(dataset, init_day)
    P_initial = copy(P)
    g = gen_square_grid(N_side)
    slack_index = Int(ceil(sample_amount/2))
    #nr_of_edges = length(flow(g,P))
    global C = JLD.load(Filename_C*".jld")["C"]
    #global C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        g = gen_square_grid(N_side)
        dataset = copy(dataset_original)
        P = initial_inject(dataset, init_day)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_C*".jld")["C"]
        #C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
        else 
            print("Capacity works good to start it!")
        end
        println("Len_P:", length(P))
        println("Len_Capacity:", length(C))
        en = Float64[ ]
        clustering_measure_list = []
        mean_P = Float64[]
        vari_P = Float64[]
        weighted_pi_sum = Float64[]
        redundant_capacity_value = Float64[]
        av_red_C = Float64[]
        #ItterStep_Drift_Enegry = Tuple{Int, Int,Float64}[]
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        N_removals = 0
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            #@time begin
            energy_old = energy!(g, P_old, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            C = copy(C_old)
            energy_new = energy!(g, P, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - energy_old[2]
            #### performance: let energy() calculate G_av only
            #end
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                     # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x < vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #slack_index = Int(ceil((length(dataset[1,:])+1)/2)) #+1 comes from P not including the slackbus node yet
                    insert!(P,slack_index,Slackbus)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                end
            end
            g = gen_square_grid(N_side)
            C = copy(C_old)
            network_measure_results = []
            for i in 1:100
                push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P, N_side, C)[2][:,1])); digits=3))
            end
            mean_network_measure = round(mean(network_measure_results); digits=3)
            std_network_measure = round(std(network_measure_results); digits=3)
            clustering_measure = (mean_network_measure, std_network_measure)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,slack_index)
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            # this gets a distance to slackbus weighted sum over all Pi's
            #dist = LightGraphs.gdistances(g,Int(ceil(sample_amount/2))) #oldversion
            #push!(weighted_pi_sum, round(sum(P.*dist), digits=3)) #oldversion
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(en, energy_old[2])
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            #### this part extracts the configuration along the way of sampling through the year, that has minimal G_av
            if isempty(lowE_config)
                #print("start")
                data_lowE = P_old, energy_old, nr_gen_con(P_old, N_side), drift_old, dataset_old, av_red_C[end]
                global locality_final = loc_1step!(P, C, N_side), loc_1step_0!(P, C, N_side)
                global P_diff_final = P_diff_all_edges(P, N_side)
                global flow_final = flow(g,P)
                global line_loadings_final = eval_line_loadings(flow_final,C)
                global Critical_links_final = eval_critical_links(g,P,C)
                push!(lowE_config, data_lowE)
            else
                if energy_old[2] <= lowE_config[1][2][2]
                    #print("update")
                    data_lowE = P_old, energy_old, nr_gen_con(P_old, N_side), drift_old, dataset_old, av_red_C[end]
                    global locality_final = loc_1step!(P, C, N_side), loc_1step_0!(P, C, N_side)
                    global P_diff_final = P_diff_all_edges(P, N_side)
                    global flow_final = flow(g,P)
                    global line_loadings_final = eval_line_loadings(flow_final,C)
                    global Critical_links_final = eval_critical_links(g,P,C)
                    lowE_config[1] = data_lowE
                else
                    lowE_config
                end
            end
            #### this part extracts the configuration along the way of sampling through the year, that is not vulnerable 
            #### but has high clustering measure.. if existing --> to check for why it is showing this clustering
            #### while not being vulnerable 
            #if isempty(nonvulnerable_highcluster_config)
            #    #print("start")
            #    data_highGav_highcluster = P_old, energy_old, clustering_measure, drift_old, dataset_old
            #    push!(nonvulnerable_highcluster_config, data_highGav_highcluster)
            #else
            #    if clustering_measure[1] >= nonvulnerable_highcluster_config[1][3][1] && energy_old[2] >= nonvulnerable_highcluster_config[1][2][2]
            #        #print("update")
            #        data_highGav_highcluster = P_old, energy_old, clustering_measure, drift_old, dataset_old
            #        nonvulnerable_highcluster_config[1] = data_highGav_highcluster
            #    else
            #        nonvulnerable_highcluster_config
            #    end
            #end
        end
        g = gen_square_grid(N_side)
        C = copy(C_old)
        energy_initial = energy!(g, P_initial, C, N_side, N_removals)
        g = gen_square_grid(N_side)
        C = copy(C_old)
        locality_init = loc_1step!(P_initial, C, N_side), loc_1step_0!(P_initial, C, N_side)
        g = gen_square_grid(N_side)
        C = copy(C_old)
        energy_final = lowE_config[1][2]
        P_diff_init = P_diff_all_edges(P_initial, N_side)
        flow_init = flow(g,P_initial)
        line_loadings_init = eval_line_loadings(flow_init,C)
        Critical_links_init = eval_critical_links(g,P_initial,C)
        g = gen_square_grid(N_side)
        C = copy(C_old)
        SA_extremal = P_initial, energy_initial, nr_gen_con(P_initial, N_side), P, energy_final, nr_gen_con(P, N_side), Drift_evol, Swap_evol, en, flow_init, flow_final, P_diff_init, P_diff_final, Critical_links_init, Critical_links_final, lowE_config, mean_P, vari_P, weighted_pi_sum, clustering_measure_list, redundant_capacity_value, locality_init, locality_final, line_loadings_init, line_loadings_final, Counter, av_red_C
        print("Capacity Violation in each MC Step:", Counter)
        push!(Data, SA_extremal)
    end 
    if saving == true
        JLD.save(Filename_output*".jld", "Data", Data, "N_side", N_side, "NV(g)", length(P_initial), "NE(g)", length(line_capacity(N_side,P_initial,buffer)), "k_max", k_max, "N_runs", N_runs, "flow_threshold", T, "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types, "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob, "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer)
    end
    Data
end

""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_new(Filename_input::String, Filename_output::String, Filename_C::String, time::String, N_runs::Int64, N_side::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, T::Float64, annealing_schedule, k_max::Int64, saving::Bool)
    Data = []
    lowE_config = []
    #nonvulnerable_highcluster_config = []
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    sample_amount = N_side * N_side
    dataset = init_injection(Filename_input, distr_of_node_types, sample_amount, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    P = initial_inject(dataset, init_day)
    P_initial = copy(P)
    g = gen_square_grid(N_side)
    slack_index = Int(ceil(sample_amount/2))
    #nr_of_edges = length(flow(g,P))
    global C = JLD.load(Filename_C*".jld")["C"]
    #global C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = gen_square_grid(N_side)
        dataset = copy(dataset_original)
        P = initial_inject(dataset, init_day)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_C*".jld")["C"]
        #C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
        else 
            print("Capacity works good to start it!")
        end
        println("Len_P:", length(P))
        println("Len_Capacity:", length(C))
        en = Float64[ ]
        clustering_measure_list = []
        mean_P = Float64[]
        vari_P = Float64[]
        weighted_pi_sum = Float64[]
        redundant_capacity_value = Float64[]
        av_red_C = Float64[]
        #ItterStep_Drift_Enegry = Tuple{Int, Int,Float64}[]
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        network_measure_results = []
        for i in 1:100
            push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P_initial, N_side, C)[2][:,1])); digits=3))
        end
        mean_network_measure = round(mean(network_measure_results); digits=3)
        std_network_measure = round(std(network_measure_results); digits=3)
        clustering_measure = (mean_network_measure, std_network_measure)
        push!(clustering_measure_list, clustering_measure)
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,slack_index)
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        energy_initial = energy!(g, P_initial, C, N_side, N_removals)
        g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        push!(en, energy_initial[2]) # before it was energy_old 25.11.2021
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        locality_init = loc_1step!(P_initial, C, N_side), loc_1step_0!(P_initial, C, N_side)
        g = gen_square_grid(N_side) # kann evtl weg
        C = copy(C_old) # kann evtl weg
        P_diff_init = P_diff_all_edges(P_initial, N_side)
        flow_init = flow(g,P_initial)
        line_loadings_init = eval_line_loadings(flow_init,C)
        Critical_links_init = eval_critical_links(g,P_initial,C)
        g = gen_square_grid(N_side) # kann evtl weg
        C = copy(C_old) # kann evtl weg
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)
        
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            energy_old = copy(energy_new)
            #@time begin
            #energy_old = energy!(g, P_old, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            energy_new = energy!(g, P, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            g = gen_square_grid(N_side)
            C = copy(C_old)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - en[end]
            #### performance: let energy() calculate G_av only
            #end
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                     # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))] 
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #slack_index = Int(ceil((length(dataset[1,:])+1)/2)) #+1 comes from P not including the slackbus node yet
                    insert!(P,slack_index,Slackbus)
                    energy_new = energy!(g, P, C, N_side, N_removals)
                    g = gen_square_grid(N_side)
                    C = copy(C_old)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            network_measure_results = []
            for i in 1:100
                push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P, N_side, C)[2][:,1])); digits=3))
            end
            mean_network_measure = round(mean(network_measure_results); digits=3)
            std_network_measure = round(std(network_measure_results); digits=3)
            clustering_measure = (mean_network_measure, std_network_measure)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,slack_index)
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            # this gets a distance to slackbus weighted sum over all Pi's
            #dist = LightGraphs.gdistances(g,Int(ceil(sample_amount/2))) #oldversion
            #push!(weighted_pi_sum, round(sum(P.*dist), digits=3)) #oldversion
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(en, energy_new[2]) # before it was energy_old 25.11.2021
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            #### this part extracts the configuration along the way of sampling through the year, that has minimal G_av
            if isempty(lowE_config)
                #print("start")
                data_lowE = P, energy_new, nr_gen_con(P, N_side), drift, dataset, av_red_C[end] #was before: P_old, energy_old, nr_gen_con(P_old, N_side), drift_old, dataset_old, av_red_C[end]
                global locality_final = loc_1step!(P, C, N_side), loc_1step_0!(P, C, N_side)
                global P_diff_final = P_diff_all_edges(P, N_side)
                global flow_final = flow(g,P)
                global line_loadings_final = eval_line_loadings(flow_final,C)
                global Critical_links_final = eval_critical_links(g,P,C)
                push!(lowE_config, data_lowE)
            else
                if energy_new[2] <= lowE_config[1][2][2]
                    #print("update")
                    data_lowE = P, energy_new, nr_gen_con(P, N_side), drift, dataset, av_red_C[end] #was before: P_old, energy_old, nr_gen_con(P_old, N_side), drift_old, dataset_old, av_red_C[end]
                    global locality_final = loc_1step!(P, C, N_side), loc_1step_0!(P, C, N_side)
                    global P_diff_final = P_diff_all_edges(P, N_side)
                    global flow_final = flow(g,P)
                    global line_loadings_final = eval_line_loadings(flow_final,C)
                    global Critical_links_final = eval_critical_links(g,P,C)
                    lowE_config[1] = data_lowE
                else
                    lowE_config
                end
            end
            #### this part extracts the configuration along the way of sampling through the year, that is not vulnerable 
            #### but has high clustering measure.. if existing --> to check for why it is showing this clustering
            #### while not being vulnerable 
            #if isempty(nonvulnerable_highcluster_config)
            #    #print("start")
            #    data_highGav_highcluster = P_old, energy_old, clustering_measure, drift_old, dataset_old
            #    push!(nonvulnerable_highcluster_config, data_highGav_highcluster)
            #else
            #    if clustering_measure[1] >= nonvulnerable_highcluster_config[1][3][1] && energy_old[2] >= nonvulnerable_highcluster_config[1][2][2]
            #        #print("update")
            #        data_highGav_highcluster = P_old, energy_old, clustering_measure, drift_old, dataset_old
            #        nonvulnerable_highcluster_config[1] = data_highGav_highcluster
            #    else
            #        nonvulnerable_highcluster_config
            #    end
            #end
        end
        energy_final = lowE_config[1][2]
        g = gen_square_grid(N_side)
        C = copy(C_old)
        SA_extremal = P_initial, energy_initial, nr_gen_con(P_initial, N_side), P, energy_final, nr_gen_con(P, N_side), Drift_evol, Swap_evol, en, flow_init, flow_final, P_diff_init, P_diff_final, Critical_links_init, Critical_links_final, lowE_config, mean_P, vari_P, weighted_pi_sum, clustering_measure_list, redundant_capacity_value, locality_init, locality_final, line_loadings_init, line_loadings_final, Counter, av_red_C
        print("Capacity Violation in each MC Step:", Counter)
        push!(Data, SA_extremal)
    end 
    if saving == true
        JLD.save(Filename_output*".jld", "Data", Data, "N_side", N_side, "NV(g)", length(P_initial), "NE(g)", length(line_capacity(N_side,P_initial,buffer)), "k_max", k_max, "N_runs", N_runs, "flow_threshold", T, "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types, "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob, "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer)
    end
    Data
end
#print("hello")
#combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift("Data_Set_OPSD", "Analysis_RUNS/Square_Grid/RUN1_zzzz_test", "Analysis_RUNS/Square_Grid/capacity_square_grid", "10:00:00", 1, 9, [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2], 3, 119, 1.0, 0.002, 4, 0.1, 0.92, 1.0, temp_ex00, 20, true)

function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_new_saveall(Filename_input::String, Filename_output::String, Filename_C::String, Filename_gen_con::String, time::String, N_runs::Int64, N_side::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, T::Float64, annealing_schedule, k_max::Int64, saving::Bool)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    sample_amount = N_side * N_side
    dataset = init_injection(Filename_input, distr_of_node_types, sample_amount, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    P = initial_inject(dataset, init_day)
    P_initial = copy(P)
    g = gen_square_grid(N_side)
    slack_index = Int(ceil(sample_amount/2))
    #nr_of_edges = length(flow(g,P))
    global C = JLD.load(Filename_C*".jld")["C"]
    #global C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = gen_square_grid(N_side)
        dataset = copy(dataset_original)
        P = initial_inject(dataset, init_day)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_C*".jld")["C"]
        #C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
        else 
            print("Capacity works good to start it!")
        end
        println("Len_P:", length(P))
        println("Len_Capacity:", length(C))
        
        en = Float64[]
        energy_initial = energy!(g, P_initial, C, N_side, N_removals)
        g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        push!(en, energy_initial[2]) # before it was energy_old 25.11.2021
        clustering_measure_list = []
        network_measure_results = []
        for i in 1:100
            push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P_initial, N_side, C)[2][:,1])); digits=3))
        end
        mean_network_measure = round(mean(network_measure_results); digits=3)
        std_network_measure = round(std(network_measure_results); digits=3)
        clustering_measure = (mean_network_measure, std_network_measure)
        push!(clustering_measure_list, clustering_measure)
        mean_P = Float64[]
        vari_P = Float64[]
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,slack_index)
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        weighted_pi_sum = Float64[]
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        redundant_capacity_value = Float64[]
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        av_red_C = Float64[]
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        Ps = Array{Float64,1}[]
        push!(Ps, P)
        energies = []
        push!(energies, energy_initial[1])
        flows = Array{Float64,1}[]
        push!(flows, flow(g,P))
        P_diffs = Array{Float64,1}[]
        push!(P_diffs, P_diff_all_edges(P, N_side))
        Critical_links = Array{Any,1}[]
        non_reroutable_flow = Array{Any,1}[]
        critical_links_result = eval_critical_links(g,P,C)
        push!(Critical_links, critical_links_result[1])
        push!(non_reroutable_flow, critical_links_result[2])
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flows[end],C))
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)
        
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            energy_old = copy(energy_new)
            #@time begin
            #energy_old = energy!(g, P_old, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            energy_new = energy!(g, P, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            g = gen_square_grid(N_side)
            C = copy(C_old)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - en[end]
            #### performance: let energy() calculate G_av only
            #end
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                     # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))] 
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #slack_index = Int(ceil((length(dataset[1,:])+1)/2)) #+1 comes from P not including the slackbus node yet
                    insert!(P,slack_index,Slackbus)
                    energy_new = energy!(g, P, C, N_side, N_removals)
                    g = gen_square_grid(N_side)
                    C = copy(C_old)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            push!(en, energy_new[2]) # before it was energy_old 25.11.2021
            network_measure_results = []
            for i in 1:100
                push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P, N_side, C)[2][:,1])); digits=3))
            end
            mean_network_measure = round(mean(network_measure_results); digits=3)
            std_network_measure = round(std(network_measure_results); digits=3)
            clustering_measure = (mean_network_measure, std_network_measure)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,slack_index)
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            # this gets a distance to slackbus weighted sum over all Pi's
            #dist = LightGraphs.gdistances(g,Int(ceil(sample_amount/2))) #oldversion
            #push!(weighted_pi_sum, round(sum(P.*dist), digits=3)) #oldversion
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(Ps, P)
            push!(energies, energy_new[1])
            push!(flows, flow(g,P))
            push!(P_diffs, P_diff_all_edges(P, N_side))
            critical_links_result = eval_critical_links(g,P,C)
            push!(Critical_links, critical_links_result[1])
            push!(non_reroutable_flow, critical_links_result[2])
            push!(line_loadings, eval_line_loadings(flows[end],C))
        end
        ######### EVAL MEASURES SECTION ##########
        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        av_nr_gen_con = zeros(length(Ps))
        nrs_gen_con = zeros(length(Ps))
        for (i,P) in enumerate(Ps)
            gen_gen, con_con, gen_con = nr_gen_con(P,N_side)
            nrs_gen_con[i] = gen_con
            nr_prod = length(findall(P .> 0))
            av_nr_gen_con[i] = G1_gen_con_edges_average[nr_prod]
        end
        clustering_measure_sum = Float64[]
        for i in 1:length(clustering_measure_list)
            push!(clustering_measure_sum, clustering_measure_list[i][1])
        end
        stable_pos = findall(x->x==maximum(en), en)
        mean_line_loadings = round.(mean.(line_loadings), digits=3)
        max_lineload_at_stable = maximum(mean_line_loadings[stable_pos])
        max_pmean_at_stable = maximum(mean_P[stable_pos])
        measures, measures_string = define_vulnerability_measures(vari_P,mean_P,max_pmean_at_stable,clustering_measure_sum,
                                    redundant_capacity_value,av_red_C,av_nr_gen_con,
                                    nrs_gen_con,mean_line_loadings,max_lineload_at_stable,weighted_pi_sum)
        FP_Data = eval_FP_for_measures(en, measures, energies)

        JLD2.save(Filename_output*"_FP_Analysis.jld2", "FP_Analysis", FP_Data, "nrs_gen_con", nrs_gen_con, "av_nr_gen_con", av_nr_gen_con, 
                    "mean_line_loadings", mean_line_loadings, "max_lineload_at_stable", max_lineload_at_stable, "max_pmean_at_stable", max_pmean_at_stable,
                    "measures", measures, "measures_string", measures_string)
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_flows.jld2", "flows", flows)
        JLD2.save(Filename_output*"_Pdiffs.jld2", "P_diffs", P_diffs)
        JLD2.save(Filename_output*"_Critical_links.jld2", "Critical_links", Critical_links)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*".jld2", "N_side", N_side, "NV(g)", length(P_initial), "NE(g)", length(line_capacity(N_side,P_initial,buffer)), 
                    "k_max", k_max, "N_runs", N_runs, "flow_threshold", T,
                    "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en, "mean_P", mean_P, "vari_P", vari_P, "weighted_pi_sum", weighted_pi_sum, "clustering_measure_list", clustering_measure_list, 
                    "redundant_capacity_value", redundant_capacity_value, "Counter", Counter, "av_red_C", av_red_C, "non_reroutable_flow", non_reroutable_flow
                    )
        print("Capacity Violation in each MC Step:", Counter)
        print("not able to redistribute flow for lines that are overloaded from the beginning:", non_reroutable_flow)
    end
end
#print("hello")
#combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_new_saveall("Data_Set_OPSD", "Analysis_RUNS_new/Square_Grid/RUN1_zzzz_test", "Analysis_RUNS_new/Square_Grid/capacity_square_grid", "Analysis_RUNS_new/Square_Grid/square_average_nrs_gen_con_edges", "10:00:00", 1, 9, [0,0,0,0,0,0.2,0.2,0.2,0.2,0.2], 3, 119, 1.0, 0.002, 4, 0.1, 0.92, 1.0, temp_ex00, 13, true)

function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_newcascading_saveall(Filename_input::String, Filename_output::String, Filename_C::String, Filename_gen_con::String, time::String, N_runs::Int64, N_side::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, T::Float64, hh_battery_capacity::Float64, slack_limit::Float64, annealing_schedule, k_max::Int64, saving::Bool)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    sample_amount = N_side * N_side
    dataset = init_injection(Filename_input, distr_of_node_types, sample_amount, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    P = initial_inject(dataset, init_day)
    P_initial = copy(P)
    g = gen_square_grid(N_side)
    slack_index = Int(ceil(sample_amount/2))
    #nr_of_edges = length(flow(g,P))
    global C = JLD.load(Filename_C*".jld")["C"]
    #global C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = gen_square_grid(N_side)
        dataset = copy(dataset_original)
        P = initial_inject(dataset, init_day)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_C*".jld")["C"]
        #C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
            flush(stdout)
        else 
            print("Capacity works good to start it!")
            flush(stdout)
        end
        println("Len_P:", length(P))
        flush(stdout)
        println("Len_Capacity:", length(C))
        flush(stdout)
        
        en = Float64[]
        #energy_initial = energy!(g, P_initial, C, N_side, N_removals)
        #energy_initial = improved_energy!(g,P,C,hh_battery_capacity)
        energy_initial = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
        g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        P = copy(P_initial)
        push!(en, energy_initial[2]) # before it was energy_old 25.11.2021
        clustering_measure_list = []
        network_measure_results = []
        for i in 1:100
            push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P_initial, N_side, C)[2][:,1])); digits=3))
        end
        mean_network_measure = round(mean(network_measure_results); digits=3)
        std_network_measure = round(std(network_measure_results); digits=3)
        clustering_measure = (mean_network_measure, std_network_measure)
        push!(clustering_measure_list, clustering_measure)
        mean_P = Float64[]
        vari_P = Float64[]
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,slack_index)
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        weighted_pi_sum = Float64[]
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        redundant_capacity_value = Float64[]
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        av_red_C = Float64[]
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        Ps = Array{Float64,1}[]
        push!(Ps, P)
        energies = []
        push!(energies, energy_initial[1])
        flows = Array{Float64,1}[]
        push!(flows, flow(g,P))
        P_diffs = Array{Float64,1}[]
        push!(P_diffs, P_diff_all_edges(P, N_side))
        Critical_links = Array{Any,1}[]
        non_reroutable_flow = Array{Any,1}[]
        critical_links_result = eval_critical_links(g,P,C)
        push!(Critical_links, critical_links_result[1])
        push!(non_reroutable_flow, critical_links_result[2])
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flows[end],C))
        max_battery_usages = Float64[]  ###battery case
        push!(max_battery_usages, energy_initial[3]) ###battery case
        av_max_battery_usage = Float64[] ###battery case
        push!(av_max_battery_usage, energy_initial[4]) ###battery case
        max_slack_usages = Float64[]  ###battery case
        push!(max_slack_usages, energy_initial[5]) ###battery case
        av_max_slack_usage = Float64[] ###battery case
        push!(av_max_slack_usage, energy_initial[6]) ###battery case
        critical_cases = Vector{Tuple{Float64,Int64}}[] ###battery case
        push!(critical_cases, energy_initial[7]) ###battery case
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)
        
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            energy_old = copy(energy_new)
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            @label new_config_gen ###battery case
            while 1 in (abs.(flow(g, P)) .> C) ###battery case
                var_at_place = copy(var_at_place_old)
                dataset = copy(dataset_old)
                drift = copy(drift_old)
                swap = copy(swap_old)
                #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
                dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            end
            P_new = copy(P)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            
            #@time begin
            #energy_old = energy!(g, P_old, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            #energy_new = energy!(g, P, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            #energy_new = improved_energy!(g, P, C, hh_battery_capacity)
            energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
            g = gen_square_grid(N_side)
            C = copy(C_old)
            P = copy(P_new)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - en[end]
            #### performance: let energy() calculate G_av only
            #end
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                     # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))] 
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #slack_index = Int(ceil((length(dataset[1,:])+1)/2)) #+1 comes from P not including the slackbus node yet
                    insert!(P,slack_index,Slackbus)
                    if 1 in (abs.(flow(g, P)) .> C) ###battery case
                        @goto new_config_gen
                    end
                    P_newtoo = copy(P)
                    #energy_new = energy!(g, P, C, N_side, N_removals)
                    #energy_new = improved_energy!(g, P, C, hh_battery_capacity)
                    energy_new = improved_energy_slacklimit!(g,P,C,hh_battery_capacity,slack_limit,slack_index)
                    g = gen_square_grid(N_side)
                    C = copy(C_old)
                    P = copy(P_newtoo)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            push!(en, energy_new[2]) # before it was energy_old 25.11.2021
            network_measure_results = []
            for i in 1:100
                push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P, N_side, C)[2][:,1])); digits=3))
            end
            mean_network_measure = round(mean(network_measure_results); digits=3)
            std_network_measure = round(std(network_measure_results); digits=3)
            clustering_measure = (mean_network_measure, std_network_measure)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,slack_index)
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            # this gets a distance to slackbus weighted sum over all Pi's
            #dist = LightGraphs.gdistances(g,Int(ceil(sample_amount/2))) #oldversion
            #push!(weighted_pi_sum, round(sum(P.*dist), digits=3)) #oldversion
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(Ps, P)
            push!(energies, energy_new[1])
            push!(flows, flow(g,P))
            push!(P_diffs, P_diff_all_edges(P, N_side))
            critical_links_result = eval_critical_links(g,P,C)
            push!(Critical_links, critical_links_result[1])
            push!(non_reroutable_flow, critical_links_result[2])
            push!(line_loadings, eval_line_loadings(flows[end],C))
            push!(max_battery_usages, energy_new[3])
            push!(av_max_battery_usage, energy_new[4])
            push!(max_slack_usages, energy_new[5])
            push!(av_max_slack_usage, energy_new[6])
            push!(critical_cases, energy_new[7])
        end
        ######### EVAL MEASURES SECTION ##########
        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        av_nr_gen_con = zeros(length(Ps))
        nrs_gen_con = zeros(length(Ps))
        for (i,P) in enumerate(Ps)
            gen_gen, con_con, gen_con = nr_gen_con(P,N_side)
            nrs_gen_con[i] = gen_con
            nr_prod = length(findall(P .> 0))
            av_nr_gen_con[i] = G1_gen_con_edges_average[nr_prod]
        end
        clustering_measure_sum = Float64[]
        for i in 1:length(clustering_measure_list)
            push!(clustering_measure_sum, clustering_measure_list[i][1])
        end
        stable_pos = findall(x->x==maximum(en), en)
        mean_line_loadings = round.(mean.(line_loadings), digits=3)
        max_lineload_at_stable = maximum(mean_line_loadings[stable_pos])
        max_pmean_at_stable = maximum(mean_P[stable_pos])
        measures, measures_string = define_vulnerability_measures(vari_P,mean_P,max_pmean_at_stable,clustering_measure_sum,
                                    redundant_capacity_value,av_red_C,av_nr_gen_con,
                                    nrs_gen_con,mean_line_loadings,max_lineload_at_stable,weighted_pi_sum)
        FP_Data = eval_FP_for_measures(en, measures, energies)

        JLD2.save(Filename_output*"_FP_Analysis.jld2", "FP_Analysis", FP_Data, "nrs_gen_con", nrs_gen_con, "av_nr_gen_con", av_nr_gen_con, 
                    "mean_line_loadings", mean_line_loadings, "max_lineload_at_stable", max_lineload_at_stable, "max_pmean_at_stable", max_pmean_at_stable,
                    "measures", measures, "measures_string", measures_string)
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_flows.jld2", "flows", flows)
        JLD2.save(Filename_output*"_Pdiffs.jld2", "P_diffs", P_diffs)
        JLD2.save(Filename_output*"_Critical_links.jld2", "Critical_links", Critical_links)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*"_battery_slack.jld2", "max_battery_usages", max_battery_usages, "av_max_battery_usage", av_max_battery_usage,
                         "max_slack_usages", max_slack_usages, "av_max_slack_usage", av_max_slack_usage, "critical_cases", critical_cases)
        JLD2.save(Filename_output*".jld2", "N_side", N_side, "NV(g)", length(P_initial), "NE(g)", length(line_capacity(N_side,P_initial,buffer)), 
                    "k_max", k_max, "N_runs", N_runs, "flow_threshold", T,
                    "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en, "mean_P", mean_P, "vari_P", vari_P, "weighted_pi_sum", weighted_pi_sum, "clustering_measure_list", clustering_measure_list, 
                    "redundant_capacity_value", redundant_capacity_value, "Counter", Counter, "av_red_C", av_red_C, "non_reroutable_flow", non_reroutable_flow
                    )
        print("Capacity Violation in each MC Step:", Counter)
        print("not able to redistribute flow for lines that are overloaded from the beginning:", non_reroutable_flow)
    end
end

function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_new_saveall_critlink(Filename_input::String, Filename_output::String, Filename_C::String, Filename_gen_con::String, time::String, N_runs::Int64, N_side::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, T::Float64, annealing_schedule, k_max::Int64, saving::Bool)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    sample_amount = N_side * N_side
    dataset = init_injection(Filename_input, distr_of_node_types, sample_amount, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    P = initial_inject(dataset, init_day)
    P_initial = copy(P)
    g = gen_square_grid(N_side)
    slack_index = Int(ceil(sample_amount/2))
    #nr_of_edges = length(flow(g,P))
    global C = JLD.load(Filename_C*".jld")["C"]
    #global C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = gen_square_grid(N_side)
        dataset = copy(dataset_original)
        P = initial_inject(dataset, init_day)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_C*".jld")["C"]
        #C = line_capacity_max_of_permutations_withweighted_limits(dataset,N_side,nr_of_edges,init_day,buffer)
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
        else 
            print("Capacity works good to start it!")
        end
        println("Len_P:", length(P))
        println("Len_Capacity:", length(C))
        
        en = Float64[]
        energy_initial = energy!(g, P_initial, C, N_side, N_removals)
        g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        push!(en, energy_initial[2]) # before it was energy_old 25.11.2021
        clustering_measure_list = []
        network_measure_results = []
        for i in 1:100
            push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P_initial, N_side, C)[2][:,1])); digits=3))
        end
        mean_network_measure = round(mean(network_measure_results); digits=3)
        std_network_measure = round(std(network_measure_results); digits=3)
        clustering_measure = (mean_network_measure, std_network_measure)
        push!(clustering_measure_list, clustering_measure)
        mean_P = Float64[]
        vari_P = Float64[]
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,slack_index)
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        weighted_pi_sum = Float64[]
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        redundant_capacity_value = Float64[]
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        av_red_C = Float64[]
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        Ps = Array{Float64,1}[]
        push!(Ps, P)
        energies = []
        push!(energies, energy_initial[1])
        flows = Array{Float64,1}[]
        push!(flows, flow(g,P))
        P_diffs = Array{Float64,1}[]
        push!(P_diffs, P_diff_all_edges(P, N_side))
        Critical_links = Array{Any,1}[]
        critlink_orig = Float64[]
        non_reroutable_flow = Array{Any,1}[]
        critical_links_result = eval_critical_links(g,P,C)
        push!(Critical_links, critical_links_result[1])
        push!(critlink_orig, round(mean(critical_links_result[1][findall(x->x!="inf", critical_links_result[1])]), digits=3))
        push!(non_reroutable_flow, critical_links_result[2])
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flows[end],C))
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        critical_links_result_new = copy(critical_links_result)
        critlink_new = copy(critlink_orig[end])
        
        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1)
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            critical_links_result_old = copy(critical_links_result_new)
            #@time begin
            #energy_old = energy!(g, P_old, C, N_side, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = gen_square_grid(N_side)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            critical_links_result_new = eval_critical_links(g,P,C)
            critlink_new = round(mean(critical_links_result_new[1][findall(x->x!="inf", critical_links_result_new[1])]), digits=3)
            ##print("I am done with energy_new")
            ΔE = (critlink_orig[end] - critlink_new) * 100.0
            #### performance: let energy() calculate G_av only
            #end
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                     # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                critical_links_result_new
                critlink_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                critical_links_result_new
                critlink_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))] 
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #slack_index = Int(ceil((length(dataset[1,:])+1)/2)) #+1 comes from P not including the slackbus node yet
                    insert!(P,slack_index,Slackbus)
                    critical_links_result_new = eval_critical_links(g,P,C)
                    critlink_new = round(mean(critical_links_result_new[1][findall(x->x!="inf", critical_links_result_new[1])]), digits=3)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    critical_links_result_new = copy(critical_links_result_old)
                    critlink_new = copy(critlink_orig[end])
                end
            end
            energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            g = gen_square_grid(N_side)
            C = copy(C_old)
            push!(en, energy_new[2]) # before it was energy_old 25.11.2021
            network_measure_results = []
            for i in 1:100
                push!(network_measure_results, round(sum(abs.(eval_clustering_network_measure_for_P(P, N_side, C)[2][:,1])); digits=3))
            end
            mean_network_measure = round(mean(network_measure_results); digits=3)
            std_network_measure = round(std(network_measure_results); digits=3)
            clustering_measure = (mean_network_measure, std_network_measure)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,slack_index)
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            # this gets a distance to slackbus weighted sum over all Pi's
            #dist = LightGraphs.gdistances(g,Int(ceil(sample_amount/2))) #oldversion
            #push!(weighted_pi_sum, round(sum(P.*dist), digits=3)) #oldversion
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(Ps, P)
            push!(energies, energy_new[1])
            push!(flows, flow(g,P))
            push!(P_diffs, P_diff_all_edges(P, N_side))
            push!(critlink_orig, round(mean(critical_links_result_new[1][findall(x->x!="inf", critical_links_result_new[1])]), digits=3))
            push!(Critical_links, critical_links_result_new[1])
            push!(non_reroutable_flow, critical_links_result_new[2])
            push!(line_loadings, eval_line_loadings(flows[end],C))
        end
        ######### EVAL MEASURES SECTION ##########
        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        av_nr_gen_con = zeros(length(Ps))
        nrs_gen_con = zeros(length(Ps))
        for (i,P) in enumerate(Ps)
            gen_gen, con_con, gen_con = nr_gen_con(P,N_side)
            nrs_gen_con[i] = gen_con
            nr_prod = length(findall(P .> 0))
            av_nr_gen_con[i] = G1_gen_con_edges_average[nr_prod]
        end
        clustering_measure_sum = Float64[]
        for i in 1:length(clustering_measure_list)
            push!(clustering_measure_sum, clustering_measure_list[i][1])
        end
        above_10_cases = zeros(length(critlink_orig))
        critlink = zeros(length(critlink_orig))
        for (i,elem) in enumerate(Critical_links)
            non_inf = elem[findall(x->x!="inf", elem)]
            extreme_case_nr = count(non_inf .> 10.0)
            above_10_cases[i] = extreme_case_nr
            critlink[i] = round(mean(vcat(non_inf[non_inf .<= 10.0], fill(10.0,extreme_case_nr))), digits=3) 
        end
        x_thr = get_X_threshold_critlink(Critical_links, critlink, 1.0)
        stable_pos = findall(x->x <= x_thr, critlink)
        mean_line_loadings = round.(mean.(line_loadings), digits=3)
        max_lineload_at_stable = maximum(mean_line_loadings[stable_pos])
        max_pmean_at_stable = maximum(mean_P[stable_pos])
        measures, measures_string = define_vulnerability_measures_critlink(vari_P,mean_P,max_pmean_at_stable,clustering_measure_sum,
                                    redundant_capacity_value,av_red_C,av_nr_gen_con,
                                    nrs_gen_con,mean_line_loadings,max_lineload_at_stable,weighted_pi_sum)
        FP_Data = eval_FP_for_measures_critlink(critlink, measures, Critical_links)

        JLD2.save(Filename_output*"_FP_Analysis.jld2", "FP_Analysis", FP_Data, "nrs_gen_con", nrs_gen_con, "av_nr_gen_con", av_nr_gen_con, 
                    "mean_line_loadings", mean_line_loadings, "max_lineload_at_stable", max_lineload_at_stable, "max_pmean_at_stable", max_pmean_at_stable,
                    "measures", measures, "measures_string", measures_string)
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_flows.jld2", "flows", flows)
        JLD2.save(Filename_output*"_Pdiffs.jld2", "P_diffs", P_diffs)
        JLD2.save(Filename_output*"_Critical_links.jld2", "Critical_links", Critical_links)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*".jld2", "N_side", N_side, "NV(g)", length(P_initial), "NE(g)", length(line_capacity(N_side,P_initial,buffer)), 
                    "k_max", k_max, "N_runs", N_runs, "flow_threshold", T, "critlink", critlink_orig,
                    "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en, "mean_P", mean_P, "vari_P", vari_P, "weighted_pi_sum", weighted_pi_sum, "clustering_measure_list", clustering_measure_list, 
                    "redundant_capacity_value", redundant_capacity_value, "Counter", Counter, "av_red_C", av_red_C, "non_reroutable_flow", non_reroutable_flow
                    )
        print("Capacity Violation in each MC Step:", Counter)
        print("not able to redistribute flow for lines that are overloaded from the beginning:", non_reroutable_flow)
    end
end


""" this function is the MCMC Simulation and outputs a file with all stored data"""
function combined_two_step_collect_data_SA_runs_var_ann_shed_new_fixeddrift_realtopology_new_saveall_singleday(Filename_input::String, Filename_output::String, Filename_graph::String, Filename_gen_con::String, time::String, N_runs::Int64, nr_vertices::Int64, distr_of_node_types::Array{Float64,1}, PVplant_scaling, init_day::Int64, swap_prob::Float64, drift_prob::Float64, vari::Int64, devi_frac::Float64, buffer, annealing_schedule, k_max::Int64, saving::Bool)
    hour_idxs = create_hour_idx_for_sample(Filename_input, time)
    dataset = init_injection(Filename_input, distr_of_node_types, nr_vertices, PVplant_scaling, hour_idxs)
    dataset_original = copy(dataset)
    #loading of fixed real topology power grid
    adj_matrix, embedding, modularity_result = load_graph_realtopology(Filename_graph)
    #cluster_nrs, largest_cluster = size(modularity_result)
    g = LightGraphs.SimpleGraph(adj_matrix)  
    slack_index = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]  
    P = initial_inject_realtopology(dataset, init_day, g, slack_index)
    P_initial = copy(P)
    nr_of_edges = LightGraphs.ne(g)#length(flow(g,P))
    global C = JLD.load(Filename_graph*"_capacity.jld")["C"]
    #defining the dataset outside the N_runs avoids effects of different topological distribution effects in simulation
    for i in 1:N_runs ####!!!!!!!!Dataset sollte für zweiten RUN wieder das ausgangs dataset sein!!!!!!!!!!!###
        N_removals = 0
        g = LightGraphs.SimpleGraph(adj_matrix)
        dataset = copy(dataset_original)
        P = initial_inject_realtopology(dataset, init_day, g, slack_index)
        P_initial = copy(P)
        drift = 0 # in each run drift starts at zero -> just the init_day defines starting point
        swap = 0 # in each run swap starts at zero -> just the init_day defines starting point
        C = JLD.load(Filename_graph*"_capacity.jld")["C"]
        C_old = copy(C)
        if false in (C .> abs.(flow(g,P_initial)))
            print("Capacity too low for initial P at initial day!")
            flush(stdout)
        else 
            print("Capacity works good to start it!")
            flush(stdout)
        end
        println("Len_P:", length(P))
        flush(stdout)
        println("Len_Capacity:", length(C))
        flush(stdout)

        en = Float64[]
        energy_initial = energy_with_battery_check!(g,P,C)
        #energy_initial = energy_realtopology!(g, P_initial, C, N_removals)
        g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
        C = copy(C_old) #C is also mutated by energy!()
        push!(en, energy_initial[2])
        clustering_measure_list = Float64[]
        clustering_measure = eval_clustering_network_measure_for_P_realtopology(P_initial, C, Filename_graph)
        push!(clustering_measure_list, clustering_measure)
        mean_P = Float64[]
        vari_P = Float64[]
        P_copy = copy(P_initial)
        P_ohne_slack = deleteat!(P_copy,Int(slack_index))
        push!(mean_P, round(mean(P_ohne_slack), digits=3))
        push!(vari_P, round(var(P_ohne_slack), digits=3))
        weighted_pi_sum = Float64[]
        push!(weighted_pi_sum, slack_centered_intensites(g,P_initial))
        redundant_capacity_value = Float64[]
        push!(redundant_capacity_value, eval_redundant_capacity_measure(P_initial,C,g))
        av_red_C = Float64[]
        push!(av_red_C, eval_average_redundant_capacity(P_initial,C,g))
        Drift_evol = Int64[]
        Swap_evol = Int64[]
        Counter = Int64[]
        push!(Drift_evol, drift)
        push!(Swap_evol, swap)
        #push!(Counter, 0)
        Ps = Array{Float64,1}[]
        push!(Ps, P)
        energies = []
        push!(energies, energy_initial[1])
        flows = Array{Float64,1}[]
        push!(flows, flow(g,P))
        P_diffs = Array{Float64,1}[]
        push!(P_diffs, P_diff_all_edges_realtopology(P, g))
        Critical_links = Array{Any,1}[]
        non_reroutable_flow = Array{Any,1}[]
        critical_links_result = eval_critical_links(g,P,C)
        push!(Critical_links, critical_links_result[1])
        push!(non_reroutable_flow, critical_links_result[2])
        line_loadings = Array{Float64,1}[]
        push!(line_loadings, eval_line_loadings(flows[end],C))
        max_battery_usages = Array{Float64,1}[]
        push!(max_battery_usages, energy_initial[3])
        av_max_battery_usage = Float64[]
        push!(av_max_battery_usage, energy_initial[4])
        var_at_place = Int.(zeros(length(dataset[init_day,:])))
        energy_new = copy(energy_initial)

        @showprogress for l in 0:k_max - 1
            #print("MC_drift_start:", drift)
            #print("MC-Step:", l)
            Temp = annealing_schedule(l)
            dataset_old = copy(dataset)
            var_at_place_old = copy(var_at_place)
            swap_old = copy(swap)
            drift_old = copy(drift)
            P_old = copy(P) # for calculating the energy of "old" configuration
            energy_old = copy(energy_new)
            ################# FOR REGULAR TIME SWAPPING ############################
            #print("I started:")
            #dataset, P, drift, swap, var_at_place = dual_swap_new_backwards_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            @label new_config_gen
            while 1 in (abs.(flow(g, P)) .> C)
                var_at_place = copy(var_at_place_old)
                dataset = copy(dataset_old)
                drift = copy(drift_old)
                swap = copy(swap_old)
                dataset, P, drift, swap, var_at_place = dual_swap_new_fixeddrift_realtopology!(dataset, init_day, swap, swap_prob, drift, drift_prob, vari, var_at_place, devi_frac, l+1, slack_index)
            end
            #dataset, P, drift, swap, var_at_place, counts = stable_dual_swapped_config_new_fixeddrift_realtopology!(g, dataset, C, init_day, swap, swap_prob, drift, drift_prob, vari::Int, var_at_place, devi_frac,l+1,slack_index)
            #print("DRIFT:", drift)  
            ##print("I found stable by swapping:", P)
            ##print("Len_P_old/C:", length(P_old),length(C))
            #@time begin
            #energy_old = en[end]#energy_realtopology!(g, P_old, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            ##print("I am done with energy_old")
            #g = LightGraphs.SimpleGraph(adj_matrix)# energy!() mutates g, so g has to be rebuilt every time before calculating energy!() again
            #C = copy(C_old)
            energy_new = energy_with_battery_check!(g,P,C)
            #energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
            g = LightGraphs.SimpleGraph(adj_matrix)
            C = copy(C_old)
            ##print("I am done with energy_new")
            ΔE = energy_new[2] - en[end]
            #end
            #### performance: let energy() calculate G_av only
            if ΔE <= 0 # man könnte conditional auch umdrehen: if (ΔE <= 0 AND probability(ΔE, T) < rand())
                                                                        # P = P_old
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            elseif probability(ΔE, Temp) > rand() # rand() gives random number element of [0,1]
                P
                var_at_place
                dataset
                swap
                drift
                energy_new
            else
                dataset = copy(dataset_old)
                swap = copy(swap_old)
                if drift != drift_old
                    #var_at_place = var_at_place_old .+ 1 # -1 to keep the old daily positions of all producers and consumers
                    #var_at_place[findall(x -> x > vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    var_at_place = var_at_place_old .- 1 # -1 to keep the old daily positions of all producers and consumers
                    var_at_place[findall(x -> x < -vari, var_at_place)] .= 0 #moving all delayed elements to new zero position (zero pos after drift +1)
                    drift #new drift gets accepted even though the energy difference of the drifted state is usually too high
                    circ_dataset = CircularArray(dataset)
                    #P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .- drift), axes(circ_dataset,2))]
                    P = circ_dataset[CartesianIndex.(Int.(var_at_place .+ init_day .+ drift), axes(circ_dataset,2))]
                    P = Array{Float64,1}(P)
                    Slackbus = sum(Float64(sum(parse.(BigFloat, string.(P))))) *  (-1)
                    #g = LightGraphs.SimpleGraph(adj_matrix)
                    #slack_pos = findall(x -> maximum(LightGraphs.closeness_centrality(g)) in x, LightGraphs.closeness_centrality(g))[1]
                    insert!(P,slack_index,Slackbus)#+1 comes from P not including the slackbus node yet
                    if 1 in (abs.(flow(g, P)) .> C)
                        @goto new_config_gen
                    end
                    energy_new = energy_with_battery_check!(g,P,C)
                    #energy_new = energy_realtopology!(g, P, C, N_removals) #[2] # by [2] only the second value of tuple is returned (G_av)
                    g = LightGraphs.SimpleGraph(adj_matrix)
                    C = copy(C_old)
                elseif drift == drift_old
                    P = copy(P_old)
                    var_at_place = copy(var_at_place_old)
                    drift ##stays drift old anyways
                    energy_new = copy(energy_old)
                end
            end
            push!(en, energy_new[2])
            clustering_measure = eval_clustering_network_measure_for_P_realtopology(P, C, Filename_graph)
            push!(clustering_measure_list, clustering_measure)
            # here we omit the slackbus value to obtain the mean and variance of just the households
            #print("slack_pos:", slack_pos)
            P_copy = copy(P)
            P_ohne_slack = deleteat!(P_copy,Int(slack_index))
            push!(mean_P, round(mean(P_ohne_slack), digits=3))
            push!(vari_P, round(var(P_ohne_slack), digits=3))
            push!(weighted_pi_sum, slack_centered_intensites(g,P))
            push!(redundant_capacity_value, eval_redundant_capacity_measure(P,C,g))
            push!(av_red_C, eval_average_redundant_capacity(P,C,g))
            push!(Drift_evol, drift)
            push!(Swap_evol, swap)
            #push!(Counter, counts)
            push!(Ps, P)
            push!(energies, energy_new[1])
            push!(flows, flow(g,P))
            push!(P_diffs, P_diff_all_edges_realtopology(P, g))
            critical_links_result = eval_critical_links(g,P,C)
            push!(Critical_links, critical_links_result[1])
            push!(non_reroutable_flow, critical_links_result[2])
            push!(line_loadings, eval_line_loadings(flows[end],C))
            push!(max_battery_usages,energy_new[3])
            push!(av_max_battery_usage, energy_new[4])
        end
        ######### EVAL MEASURES SECTION ##########
        G1_gen_con_edges_average = JLD2.load(Filename_gen_con*".jld2")["nrs_gen_con_edges"]
        av_nr_gen_con = zeros(length(Ps))
        nrs_gen_con = zeros(length(Ps))
        for (i,P) in enumerate(Ps)
            gen_gen, con_con, gen_con = nr_gen_con_realtopology(P,g)
            nrs_gen_con[i] = gen_con
            nr_prod = length(findall(P .> 0))
            av_nr_gen_con[i] = G1_gen_con_edges_average[nr_prod]
        end
        stable_pos = findall(x->x==maximum(en), en)
        mean_line_loadings = round.(mean.(line_loadings), digits=3)
        max_lineload_at_stable = maximum(mean_line_loadings[stable_pos])
        max_pmean_at_stable = maximum(mean_P[stable_pos])
        measures, measures_string = define_vulnerability_measures(vari_P,mean_P,max_pmean_at_stable,clustering_measure_list,
                                    redundant_capacity_value,av_red_C,av_nr_gen_con,
                                    nrs_gen_con,mean_line_loadings,max_lineload_at_stable,weighted_pi_sum)
        FP_Data = eval_FP_for_measures(en, measures, energies)

        JLD2.save(Filename_output*"_FP_Analysis.jld2", "FP_Analysis", FP_Data, "nrs_gen_con", nrs_gen_con, "av_nr_gen_con", av_nr_gen_con, 
                    "mean_line_loadings", mean_line_loadings, "max_lineload_at_stable", max_lineload_at_stable, "max_pmean_at_stable", max_pmean_at_stable,
                    "measures", measures, "measures_string", measures_string)
        JLD2.save(Filename_output*"_Ps.jld2", "Ps", Ps)
        JLD2.save(Filename_output*"_energies.jld2", "energies", energies)
        JLD2.save(Filename_output*"_flows.jld2", "flows", flows)
        JLD2.save(Filename_output*"_Pdiffs.jld2", "P_diffs", P_diffs)
        JLD2.save(Filename_output*"_Critical_links.jld2", "Critical_links", Critical_links)
        JLD2.save(Filename_output*"_lineloadings.jld2", "line_loadings", line_loadings)
        JLD2.save(Filename_output*"_battery.jld2", "max_battery_usages", max_battery_usages, "av_max_battery_usage", av_max_battery_usage)
        JLD2.save(Filename_output*".jld2", "Nr_Vertices", nr_vertices, "NE(g)", nr_of_edges, "k_max", k_max, "N_runs", N_runs, 
                "annealing_schedule", string(annealing_schedule), "C", C, "time", time, "distr_of_node_types", distr_of_node_types,
                    "PVplant_scaling", PVplant_scaling, "init_day", init_day, "swap_probability", swap_prob, "drift_probability", drift_prob,
                    "variance", vari, "deviation_fraction", devi_frac, "line capacity buffer", buffer, "Drift_evol", Drift_evol, "Swap_evol", Swap_evol,
                    "en", en, "mean_P", mean_P, "vari_P", vari_P, "weighted_pi_sum", weighted_pi_sum, "clustering_measure_list", clustering_measure_list, 
                    "redundant_capacity_value", redundant_capacity_value, "Counter", Counter, "av_red_C", av_red_C, "non_reroutable_flow", non_reroutable_flow
                    )
        print("Capacity Violation in each MC Step:", Counter)
        print("not able to redistribute flow for lines that are overloaded from the beginning:", non_reroutable_flow)
    end 
end
