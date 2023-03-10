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

"""realization of an initial linefailure -> reoval of edge i. Effects both the graph and the capacity array"""
function linefailure!(g, i::Int64, C::Array{Float64,1}) # edge i is removed
    B = Array(LightGraphs.incidence_matrix(g, oriented=true))
    LightGraphs.rem_edge!(g, findfirst(isodd, B[:, i]), findlast(isodd, B[:, i])) #goes in colum of incidence_mat where each col is representing one edge has two entries
    deleteat!(C,i)
    return g, C
end

function accounting_for_possible_strongly_imbalanced_components_slacklimit!(g, P::Array{Float64,1}, P_origin::Array{Float64,1}, C::Array{Float64,1}, hh_battery_capacity::Float64, slack_limit::Float64, slack_idx::Int64, connect_components_old::Array{Array{Int64,1}},max_momentan_battery_usage::Array{Float64,1},max_momentan_slack_usage::Array{Float64,1},critical_cases)
    connect_components = LightGraphs.connected_components(g)
    new_components = connect_components[connect_components .∉ Ref(connect_components_old)]
    relevant_new_components = new_components[length.(new_components) .> 1]
    nr_relevant_new_components = length(relevant_new_components)
    # identify if new components have currently still P-reduced hh --> set them back to their P_original
    new_but_Preduced = relevant_new_components[[P[relevant_new_components[i]] != P_origin[relevant_new_components[i]] for i in 1:nr_relevant_new_components]]
    [P[new_but_Preduced[i]] = P_origin[new_but_Preduced[i]] for i in 1:length(new_but_Preduced)]
    P[slack_idx] = P_origin[slack_idx]
    #actual_slack_usage = 0.0
    if nr_relevant_new_components != 0 
        #print("NEW-COMPs:", new_components)
        #print("rel_NEW-COMPs:", relevant_new_components)
        components_surplus_minus = zeros(nr_relevant_new_components)
        [components_surplus_minus[i] = sum(P[relevant_new_components[i]])/length(relevant_new_components[i]) for i in 1:nr_relevant_new_components]
        component_survive = abs.(components_surplus_minus) .< hh_battery_capacity
        exceeded_comps = findall(x->x==false, component_survive)
        nonexceeded_comps = findall(x->x==true, component_survive)
        slack_exist_bool = [slack_idx in relevant_new_components[i] for i in 1:nr_relevant_new_components]
        if true in slack_exist_bool #gibt es den slack überhaupt in einer der neu aufgetretenen komponenten
            slack_comp_nr = argmax(slack_exist_bool)
            slack_comp_val = components_surplus_minus[slack_comp_nr]
            slack_comp_nodes = relevant_new_components[slack_comp_nr]
            slack_comp_len = length(slack_comp_nodes)
            #definiere hier direkt tracking von bat und slack für slack_comp_nr
            if slack_comp_nr in exceeded_comps
                filter!(e->e ≠ slack_comp_nr, exceeded_comps) #wir betrachten slack hier und nicht mehr später
                #push!(max_momentan_battery_usage,hh_battery_capacity)
                if slack_comp_val < 0.0 #critical exceeded
                    actual_slack_usage = abs(slack_comp_val*slack_comp_len + (slack_comp_len-1)*hh_battery_capacity)
                    #push!(max_momentan_slack_usage,actual_slack_usage)
                    if actual_slack_usage <= slack_limit
                        P[slack_idx] = round(P[slack_idx] + abs(slack_comp_len*(slack_comp_val + hh_battery_capacity)), digits=3)
                        push!(max_momentan_battery_usage,hh_battery_capacity)
                        push!(max_momentan_slack_usage,actual_slack_usage)
                    else #delete all line in this component
                        push!(critical_cases,(slack_comp_val,slack_comp_len))
                        alledges = collect(edges(g))
                        delete_idxs = Int64[]
                        for node in slack_comp_nodes
                            neighbors_for_node = copy(neighbors(g,node))
                            [LightGraphs.rem_edge!(g,node,neighbor) for neighbor in neighbors_for_node]
                            delete_idxs = vcat(delete_idxs,findall(e -> e.src == node, alledges))
                        end
                        deleteat!(C,sort!(delete_idxs))
                    end
                else #slack component is net positive and critical exceeded
                    actual_slack_usage = abs(slack_comp_val*slack_comp_len - (slack_comp_len-1)*hh_battery_capacity)
                    #push!(max_momentan_slack_usage,actual_slack_usage)
                    if actual_slack_usage <= slack_limit
                        P[slack_idx] = round(P[slack_idx] - abs(slack_comp_len*(slack_comp_val - hh_battery_capacity)), digits=3)
                        push!(max_momentan_battery_usage,hh_battery_capacity)
                        push!(max_momentan_slack_usage,actual_slack_usage)
                    else #delete all line in this component
                        push!(critical_cases,(slack_comp_val,slack_comp_len))
                        alledges = collect(edges(g))
                        delete_idxs = Int64[]
                        for node in slack_comp_nodes
                            neighbors_for_node = copy(neighbors(g,node))
                            [LightGraphs.rem_edge!(g,node,neighbor) for neighbor in neighbors_for_node]
                            delete_idxs = vcat(delete_idxs,findall(e -> e.src == node, alledges))
                        end
                        deleteat!(C,sort!(delete_idxs))
                    end
                end
            elseif slack_comp_nr in nonexceeded_comps && slack_limit < abs(slack_comp_val) # slack has lower capa than batteries -> comp kann doch überladen sein
                filter!(e->e ≠ slack_comp_nr, nonexceeded_comps) #wir betrachten slack hier und nicht mehr später
                if slack_comp_val < 0.0
                    new_slack_comp_val = abs(slack_comp_val) + ((abs(slack_comp_val) - slack_limit)/(slack_comp_len-1))
                    if new_slack_comp_val > hh_battery_capacity #delete all line in this component
                        push!(critical_cases,(slack_comp_val,slack_comp_len))
                        alledges = collect(edges(g))
                        delete_idxs = Int64[]
                        for node in slack_comp_nodes
                            neighbors_for_node = copy(neighbors(g,node))
                            [LightGraphs.rem_edge!(g,node,neighbor) for neighbor in neighbors_for_node]
                            delete_idxs = vcat(delete_idxs,findall(e -> e.src == node, alledges))
                        end
                        deleteat!(C,sort!(delete_idxs))
                    else
                        push!(max_momentan_battery_usage,abs(new_slack_comp_val))
                        push!(max_momentan_slack_usage,abs(slack_limit))
                    end
                else
                    new_slack_comp_val = slack_comp_val + (slack_comp_val - slack_limit)/(slack_comp_len-1)
                    if new_slack_comp_val > hh_battery_capacity #delete all line in this component
                        push!(critical_cases,(slack_comp_val,slack_comp_len))
                        alledges = collect(edges(g))
                        delete_idxs = Int64[]
                        for node in slack_comp_nodes
                            neighbors_for_node = copy(neighbors(g,node))
                            [LightGraphs.rem_edge!(g,node,neighbor) for neighbor in neighbors_for_node]
                            delete_idxs = vcat(delete_idxs,findall(e -> e.src == node, alledges))
                        end
                        deleteat!(C,sort!(delete_idxs))
                    else
                        push!(max_momentan_battery_usage,abs(new_slack_comp_val))
                        push!(max_momentan_slack_usage,abs(slack_limit))
                    end
                end
            elseif slack_comp_nr in nonexceeded_comps && slack_limit > abs(slack_comp_val)#slack is in nonexceeded comps
                filter!(e->e ≠ slack_comp_nr, nonexceeded_comps) #wir betrachten slack hier und nicht mehr später
                push!(max_momentan_battery_usage,abs(slack_comp_val))
                push!(max_momentan_slack_usage,abs(slack_comp_val))
            end
        end
        for component_nr in nonexceeded_comps
            #stable_max = maximum([abs(minimum(components_surplus_minus[nonexceeded_comps])),abs(maximum(components_surplus_minus[nonexceeded_comps]))])
            push!(max_momentan_battery_usage,abs(components_surplus_minus[component_nr]))
        end
        for component_nr in exceeded_comps
            push!(critical_cases,(abs(components_surplus_minus[component_nr]),length(relevant_new_components[component_nr])))
            #push!(max_momentan_battery_usage,hh_battery_capacity)
            #delete all line in this component
            alledges = collect(LightGraphs.edges(g))
            delete_idxs = Int64[]
            for node in relevant_new_components[component_nr]
                neighbors_for_node = copy(neighbors(g,node))
                [LightGraphs.rem_edge!(g,node,neighbor) for neighbor in neighbors_for_node]
                delete_idxs = vcat(delete_idxs,findall(e -> e.src == node, alledges))
            end
            deleteat!(C,sort!(delete_idxs))
        end
    end
    return g, P, C, max_momentan_battery_usage, max_momentan_slack_usage, critical_cases
end

function improved_cascading_slacklimit!(g, P::Array{Float64,1}, C::Array{Float64,1}, hh_battery_capacity::Float64, slack_limit::Float64, slack_idx::Int64, linefailure_idx::Int64, max_momentan_battery_usage::Array{Float64,1}, max_momentan_slack_usage::Array{Float64,1},critical_cases,counter_val)
    #,slack_pos::Int64 , slack_buffer::Float64
    P_origin = copy(P)
    connect_components_old = LightGraphs.connected_components(g)
    g, C = linefailure!(g, linefailure_idx, C)
    g, P, C, max_momentan_battery_usage, max_momentan_slack_usage, critical_cases = accounting_for_possible_strongly_imbalanced_components_slacklimit!(g,P,P_origin,C,hh_battery_capacity,slack_limit,slack_idx,connect_components_old,max_momentan_battery_usage,max_momentan_slack_usage,critical_cases)
    F = flow(g, P)
    counter_val += 1
    while 1 in (abs.(F) .> C)
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
        F = flow(g, P)
        counter_val += 1
    end
    return g, max_momentan_battery_usage, max_momentan_slack_usage, critical_cases, counter_val
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

#include("graph_based_clustering.jl")
#include("MCMC_simulation_functions.jl")
#include("eval_measures.jl")
#include("square_grid_kmeans_clustering.jl")
#include("capacity_setting.jl")
#include("vertex_locs_and_colour_fillc.jl")
#include("new_cascading_algorithm.jl")
#include("false_positive_and_negative.jl")
#include("deprecated_unused_functions.jl")
