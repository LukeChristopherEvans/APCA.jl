module APCA
# Write your package code here.
using DataFrames
using Wavelets
using StatsBase
using ShiftedArrays

function runAPCA(oC::Vector, M::Int)
     #K. Chakrabarti et al.
     #Table V. An Algorithm to Produce the APCA
     #Algorithm Compute APCA(C,M )2
     #begin
     #1. if length(C) is not a power of two, pad it with zeros to make it so.
     #2. Perform the Haar Discrete Wavelet Transform on C.
     #3. Sort coefficients in order of decreasing normalized magnitude, truncate after M.
     #4. Reconstruct approximation (APCA representation) of C from retained coeffs.
     #5. If C was padded with zeros, truncate it to the original length.
     #6. Replace approximate segment mean values with exact mean values.
     #7. while the number of segments is greater than M
     #8. Merge the pair of segments that can be merged with least rise in error
     #9. endwhile
     #end
    # test values
    #M  = 3
    #oC = [7.,5.,5.,3.,3.,3.,4.,6.] # example in paper
    #oC = vcat(rand(40).+3,rand(40).+2,rand(40).-1)

    # 1. If length of oC is not a power of two, pad it with zeros
    newexp = ceil(log(size(oC)[1])/ log(2) )
    padsize = Int32(2^newexp - size(oC)[1])
    oCp = vcat(oC,fill(0.,padsize))
    fvlength = log2(size(oCp)[1]) # length power

    # 2. Perform the Haar Discrete Wavelet Transform on C.
    Haar = dwt(oCp,wavelet(WT.haar))

    #  Transform into weighting and structure in paper
    transHaar = []
    for i in [1:1:Int32(fvlength);]
        j = i-1
        addvec = -Haar[2^j+1:2^i] ./ sqrt(2^(fvlength - i + 1)) # get diff coeffs
        addvec2 = addvec ./ 2^((i-1)/2) # normalize
        append!(transHaar,addvec2)
    end
    #add final mean to normalized and non-normalized
    transHaar=append!(first(Haar,1)/sqrt(2^fvlength),transHaar)

    #3. Sort coefficients in order of decreasing normalized magnitude, truncate after M
    # go to float32 because of rounding
    if size(transHaar)[1] > M
        inds = sort(DataFrame(t=abs.(Float32.(transHaar)), i=1:length(transHaar)),:t,rev=true)[!,:i][1:M,:]
    else
        inds = sort(DataFrame(t=abs.(Float32.(transHaar)), i=1:length(transHaar)),:t,rev=true)[!,:i]
    end
    # get non-normalized coefs
    #4. Reconstruct approximation (APCA representation) of C from retained coeffs.
    Haar[Not(inds)] .= 0
    newrep =  idwt(Haar,wavelet(WT.haar))

    #5. If C was padded with zeros, truncate it to the original length.
    newrep = newrep[1:size(oC)[1]]

    #6. Replace approximate segment mean values with exact mean values.
    repeatlengths = vcat(1,rle(newrep)[2]) # get repeat lengths

    exactrep=[]
    for k in 2:size(repeatlengths)[1]
        s =sum(repeatlengths[1:k-1])
        e =repeatlengths[k] + s - 1
        um =mean(oCp[s:e])

        append!(exactrep,fill(um,repeatlengths[k]))
    end

    #7. while the number of segments is greater than M
    while size(rle(exactrep)[2])[1] > M
        z=vcat(rle(exactrep)[1] .- lag(rle(exactrep)[1]))
        replace!(z, missing =>-999.)
        z=abs.(convert(Vector{Float64}, z))
        l=findfirst(minimum(z) .== z)

        newreplengths = vcat(1,rle(exactrep)[2])
        exactrep
        s =sum(newreplengths[1:l-1])
        e =newreplengths[l]+newreplengths[l+1] + s - 1
        um =mean(exactrep[s:e])
        exactrep[s:e] .= um

    end

    return exactrep

  end
export runAPCA

end
