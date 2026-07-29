function load = assemble_force_newmark(epsilon,FT,FA,fa,ft,M)
load = M*((FT.*ft) + (FA.*fa));
load = epsilon*load;
end