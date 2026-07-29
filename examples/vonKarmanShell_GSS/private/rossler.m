function dy = rossler(t,y)
Np = numel(y)/3;
a = 0.2;
b = 0.2;
c = 5.7;
dy(1:Np,1) = -y(Np+1:2*Np,1)-y(2*Np+1:3*Np,1);
dy(Np+1:2*Np,1) = y(1:Np,1)+a*y(Np+1:2*Np,1);
dy(2*Np+1:3*Np,1) = b + y(2*Np+1:3*Np,1)*(y(1:Np,1)-c);

end