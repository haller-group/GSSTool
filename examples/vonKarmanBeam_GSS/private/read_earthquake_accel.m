function [TV,TH,tspan] = read_earthquake_accel(stringV,stringH,dtt,g)

TV = readtable(stringV);%'RSN28_PARKF_VER.csv'
TV = TV{:,:};
TV = TV.';
TV = g.*TV(:);
po = isnan(TV);
TV(po) = [];
TH = readtable(stringH);% 'RSN28_PARKF_HOR1.csv'
TH = TH{:,:};
TH = TH.';
TH = g.*TH(:);
po = isnan(TH);
TH(po) = [];
if max(size(TV))< max(size(TH))
    TH = TH(1:max(size(TV)));
    tspan = 0:dtt:(max(size(TV))*dtt-dtt);
else 
    TV = TV(1:max(size(TH)));
    tspan = 0:dtt:(max(size(TH))*dtt-dtt);
end    
end