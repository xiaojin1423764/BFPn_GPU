% compute the Boltzmann-Fokker-Planck equation
% the proton therapy
% compute the Boltzmann-Fokker-Planck equation
% the proton therapy
function scale_BFPn
clear;
close all;
%% setting initial condition
clear;
close all;
%% setting initial condition
Lx = 40; 
Ly = 1; 
Lz = 1;
Lu = 1;
Lv = 1;
Lg = 259;
Nx = 4000;
dt = Lx/Nx;
Nmu= 11;
Nom=11;
Ng =500;
Ny = 10 ; 
Nz =10 ; 
dy = Ly/Ny; 
dz = Lz/Nz;
dg=Lg/Ng;
du=Lu/(Nmu-1);dv=Lv/(Nom-1);
y        = (0:1:Ny)'*dy-1;
z         = (0:1:Nz)'*dz-1;
 en         =(0:1:Ng)'*dg+1;
  mu = load('C:\Users\Administrator\Desktop\proton\data_cross.txt');
  sigma_c = load('C:\Users\Administrator\Desktop\proton\cross_total.txt');
  sigma_c=0.9*sigma_c;
 aw=0.5:0.5:21.5;  sigma_c(402:444)=sigma_c(402:444)-0.0002*aw';
u         =(-0.5:du:0.5)';
v         =(-0.5:dv:0.5)';
yhalf    =(y(1:Ny)+y(2:Ny+1))/2;
zhalf    =(z(1:Nz)+z(2:Nz+1))/2;
enhalf   =(en(1:Ng)+en(2:Ng+1))/2;
tFinal = 40;
kk1 = 0;
%% initial condition
a_1=1/2/0.1/0.1;
a_2=1/2/10^(-6)/10^(-6);
% A_a=1.342641;
sig_E=1;
C_c=1/sqrt(2*pi*0.1*10^(-6));
for j = 1 : Nmu
    Omend((j-1)*(Nom)+1:(j)*(Nom),1)=-v(Nmu-j+1);
    Omend((j-1)*(Nom)+1:(j)*(Nom),2)=-u(1:Nom);
end
for k = 1: (Nmu)*(Nom)
    for i = 1: Ny+1
        for j = 1: Nz+1
            f_1(i,j,k)=exp(-(a_1*y(i)^2+a_2*Omend(k,1)^2))*exp(-(a_1*z(j)^2+a_2*Omend(k,2)^2));
        end
    end
end
for ek = 1 : Ng
    f_2(ek,1)=1/sqrt(2*pi)/sig_E*exp(-((enhalf(ek)-230)/sig_E/sqrt(2))^2);
end
f_2=sparse(f_2);
f_3=0*f_2;f_4=0*f_2;
sig_trg=cross(3,Ng,enhalf)/1.45;
S_s=stopping(en);
T_c=stra(Ng,enhalf);
% f_1 = permute(f_1,[2 1 3]);
f_1=sparse(reshape(f_1,[],Nmu*Nom));
F=kron(f_2,f_1);f_F=kron(f_3,f_1);
F1=0*kron(f_2,f_1);f_F1=0*kron(f_3,f_1);
F_tot = F+F1; f_Ftot=f_F+f_F1;

%% prepare
for j = 1 : Ng-1
 for i = 1: Ng-1
if mu(j,3) == 0
    ker_e (i,j) = 0;
elseif i<j
     ker_e (i,j) = 0;
else ker_e (i,j) = dg/mu(j,3)*exp(-(i-j+1/2)*dg/mu(j,3));
end
end
end
temp = ker_e;
for j =1:Ng-1
    for i = 1 : Ng
if i == j
ker_e(i,j) = 0;
elseif i<j
ker_e(i,j) = temp(i,j);
else
    ker_e(i,j) = temp(i-1,j);
end
 end
end
ker_e(1:Ng,Ng)=0;
tempp=ker_e;
for j =1:Ng
    for i = 1 : Ng
ker_e(j,i)=tempp(Ng-j+1,Ng-i+1);
 end
end
ker_e(ker_e<10^-5)=0;
ker_e=ker_e.*kron(sigma_c',ones(Ng,1));
ker_e=sparse(ker_e);
% temp_ke = ker_e(1:Ng-1,:)/2+ker_e(2:Ng,:)/2;
ker_eh(1:Ng,:)=ker_e;ker_eh(Ng+1,:)=0*ker_e(Ng-1,:);
ker_e1=ker_eh(1:Ng,:)/2+ker_eh(2:Ng+1,:)/2;ker_e2=-ker_eh(1:Ng,:)/2+ker_eh(2:Ng+1,:)/2;
% a = kron(ker_e,ones(Nmu*Nom,1));
for kk = 1: Nmu*Nom
% temp1(:,kk)=acosd(max((ones(Nmu*Nom,1)+Omend(:,1)*Omend(kk,1)+Omend(:,2)*Omend(kk,1))./(sqrt(ones(Nmu*Nom,1)+Omend(:,1).^2+Omend(:,2).^2).*sqrt(ones(Nmu*Nom,1)*(1+Omend(kk,1)^2+Omend(kk,2)^2))),ones(Nmu*Nom,1)));
temp1(:,kk)=abs(acosd((ones(Nmu*Nom,1)+Omend(:,1)*Omend(kk,1)+Omend(:,2)*Omend(kk,2))./(sqrt(ones(Nmu*Nom,1)+Omend(:,1).^2+Omend(:,2).^2).*sqrt(ones(Nmu*Nom,1)*(1+Omend(kk,1)^2+Omend(kk,2)^2)))));
end
for kk2 = 1 : Ng-1
if mu(kk2,1)==0
ker_v((kk2-1)*Nom*Nmu+1:(kk2)*Nom*Nmu,:)=0*ones(Nom*Nmu,Nom*Nmu);
continue
else n_n=temp1.^(mu(kk2,1)-ones(Nmu*Nom,Nmu*Nom,1)).*exp(-1/mu(kk2,2)*temp1).*((ones(Nmu*Nom,Nmu*Nom)+kron(ones(1,Nmu*Nom),Omend(1:Nmu*Nom,1)).^2+kron(ones(1,Nmu*Nom),Omend(1:Nmu*Nom,2)).^2)).^(-3/2);
n_n(find(isnan(n_n)==1)) = 0;n_n(find(isinf(n_n)==1)) = 0;n_n1=sum(n_n);tt=n_n./kron(n_n1,ones(Nmu*Nom,1));tt(find(isnan(tt)==1)) = 0;tt(find(isinf(tt)==1)) = 0;
ker_v((kk2-1)*Nom*Nmu+1:(kk2)*Nom*Nmu,:)=tt;
ker_v(ker_v<10^-3)=0;

% N_n(i,j)=1/n_n;
% ker(i,j,k)=N_n(i,j,k)*(acosd((1+Omend(kk1,1)*u(i)+Omend(kk1,2)*v(j))./(sqrt(1+Omend(kk1,1).^2+Omend(kk1,2).^2).*sqrt((1+u(i)^2+v(j)^2)) )-0.0001)).^(mu(kk2,1)-1).*exp(-1/mu(kk2,2)*(acosd((1+Omend(kk1,1)*u(i)+Omend(kk1,2)*v(j))./(sqrt(1+Omend(kk1,1).^2+Omend(kk1,2).^2).*sqrt((1+u(i)^2+v(j)^2))  )-0.0001 ))).*((1+u(i).^2+v(j).^2)).^(-3/2);
% end
end
end
 ker_v((Ng-1)*Nom*Nmu+1:(Ng)*Nom*Nmu,:)=0*ones(Nom*Nmu,Nom*Nmu);
 ker_v=sparse(ker_v);
ky=2*pi/(Ny+1)*[-(Ny+1)/2:(Ny+1)/2-1];
ky=fftshift(ky');
kz=ky;
for i =  1:Nz+1
    for j =1:Ny+1
        ytemp(i,j)=ky(j);
        ztemp(i,j)=kz(i);
    end
end
yz(:,1)=reshape(ytemp(:,:)',(Ny+1)*(Nz+1),1);
yz(:,2)=reshape(ztemp(:,:)',(Ny+1)*(Nz+1),1);
for j =  1:Nom
    for i =1:Nmu
        utemp(i,j)=2*cos(pi*(j-1)/(Nom))-2;
        vtemp(i,j)=2*cos(pi*(i-1)/(Nmu))-2;
    end
end
uuv(:,1)=reshape(utemp(:,:)',(Nom)*(Nmu),1);
uuv(:,2)=reshape(vtemp(:,:)',(Nom)*(Nmu),1);
fin=zeros(Nx+1,1);
for i = 1 : (Ny+1)*(Nz+1)
F_temp ((i-1)*Ng+1:i*Ng,:)=F(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:);
f_Ftemp ((i-1)*Ng+1:i*Ng,:)=f_F(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:);
end
F_temp=reshape(F_temp',(Ny+1)*(Nz+1),[]);f_Ftemp=reshape(f_Ftemp',(Ny+1)*(Nz+1),[]);
for ek = 1 : Ng

F_tempp(:,(ek-1)*(Nom)*(Nmu)+1:(ek)*(Nom)*(Nmu))=F_temp*(kron((ker_e1(ek,:))',ones(Nmu*Nom,Nmu*Nom)).*ker_v);
f_Ftempp(:,(ek-1)*(Nom)*(Nmu)+1:(ek)*(Nom)*(Nmu))=f_Ftemp*(kron((ker_e2(ek,:))',ones(Nmu*Nom,Nmu*Nom)).*ker_v);
end
F_temp=F_tempp;f_Ftemp=f_Ftempp;
% F_temp=F_temp*(kron(ker_e1',ones(Nmu*Nom,Nmu*Nom)).*kron(ker_v,ones(1,Ng)));f_Ftemp=f_Ftemp*(kron(ker_e2',ones(Nmu*Nom,Nmu*Nom)).*kron(ker_v,ones(1,Ng)));
F_temp=reshape(F_temp',[],(Nmu)*(Nom));f_Ftemp=reshape(f_Ftemp',[],(Nmu)*(Nom));
F_temp1=F_temp*dg;f_Ftemp1=f_Ftemp/3*dg;
for i = 1 : (Ny+1)*(Nz+1)
F_tempf(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:)=F_temp1((i-1)*Ng+1:i*Ng,:);
f_Ftempf(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:)=f_Ftemp1((i-1)*Ng+1:i*Ng,:);
end
%% main
t = 0;
kplot = 1;
clock = 1; 
while t<tFinal
%% initial integral



%% calculate primary proton
for ek = 1 : Ng
temp_1=F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
temp_2=f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
if all (temp_1(:)==0)&&all (temp_2(:)==0)
continue
else 
temp_1=full(temp_1);temp_2=full(temp_2);
temp_1=reshape(temp_1',Nom,Nmu,[]);temp_2=reshape(temp_2',Nom,Nmu,[]);
temp_6=kron(ones((Ny+1)*(Nz+1),1)/du/du/4*sig_trg(ek),utemp);temp_7=kron(ones((Ny+1)*(Nz+1),1)/dv/dv/4*sig_trg(ek),vtemp);

temp_8=(2/dt*ones((Ny+1)*(Nz+1)*Nom,Nmu)+temp_6+temp_7)./(2/dt*ones((Ny+1)*(Nz+1)*Nom,Nmu)-temp_6-temp_7);
temp_8=permute(reshape(temp_8',Nmu,Nom,[]),[2 1 3]);
for or=1:(Ny+1)*(Nz+1)
temp_1(:,:,or)=fft2(temp_1(:,:,or));
temp_2(:,:,or)=fft2(temp_2(:,:,or));
end
temp_1=temp_8.*temp_1;temp_2=temp_8.*temp_2;
for or=1:(Ny+1)*(Nz+1)
temp_1(:,:,or)=abs(ifft2(temp_1(:,:,or)));
temp_2(:,:,or)=abs(ifft2(temp_2(:,:,or)));
end
temp_1=reshape(temp_1,[],(Ny+1)*(Nz+1));temp_2=reshape(temp_2,[],(Ny+1)*(Nz+1));
% temp_1(temp_1<10^(-10))=0;temp_2(temp_2<10^(-10))=0;
temp_1=sparse(temp_1');temp_2=sparse(temp_2');
F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_1;
f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_2;
end
end

for ek = 1 : Ng
temp_1=F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
temp_2=f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
if all (temp_1(:)==0)&&all (temp_2(:)==0)
continue
else 
temp_1=full(temp_1);temp_2=full(temp_2);
temp_1=reshape(temp_1,Ny+1,Nz+1,[]);temp_2=reshape(temp_2,Ny+1,Nz+1,[]);
temp_3=kron(Omend(:,1)/2,sqrt(-1)*ytemp);temp_4=kron(Omend(:,2)/2,sqrt(-1)*ztemp);
temp_5=(2/dt*ones((Ny+1)*Nmu*Nom,Nz+1)-temp_3-temp_4)./(2/dt*ones((Ny+1)*Nmu*Nom,Nz+1)+temp_3+temp_4);
temp_5=permute(reshape(temp_5',Ny+1,Nz+1,[]),[2 1 3]);
for ord=1:Nmu*Nom
temp_1(:,:,ord)=fft2(temp_1(:,:,ord));
temp_2(:,:,ord)=fft2(temp_2(:,:,ord));
end
temp_1=temp_5.*temp_1;temp_2=temp_5.*temp_2;
for ord=1:Nmu*Nom
temp_1(:,:,ord)=abs(ifft2(temp_1(:,:,ord)));
temp_2(:,:,ord)=abs(ifft2(temp_2(:,:,ord)));
end
temp_1=reshape(temp_1,[],Nmu*Nom);temp_2=reshape(temp_2,[],Nmu*Nom);
% temp_1(temp_1<10^(-10))=0;temp_2(temp_2<10^(-10))=0;
temp_1=sparse(temp_1);temp_2=sparse(temp_2);
F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_1;
f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_2;
end
end

% Rhs_1e=f_F/dt-kron(sigma_c,ones((Ny+1)*(Nz+1),Nmu*Nom))/2.*f_F;
% Rhs_2e(1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron(3/2/dg*S_s(2:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     +kron(3/2/dg*S_s(1:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F(1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron(3/2/dg*(S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron(1/2/dg*(-S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F(1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_2e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=kron(3/2/dg*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron(3/2/dg*(S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron(1/2/dg*(-S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs_3e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2.*sigma_c(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     +kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_3e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F(1:(Ny+1)*(Nz+1)*(2-1),:))-kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2.*sigma_c(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:))...
%     +kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:));
% Rhs_3e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2.*sigma_c(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     +kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% 
% 
% Rhs_4e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2.*sigma_c(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     +kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_4e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F(1:(Ny+1)*(Nz+1)*(2-1),:))-kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2*sigma_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:))...
%     +kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:));
% Rhs_4e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2*sigma_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     +kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs_e = Rhs_1e+Rhs_2e-Rhs_3e;
% for ek = Ng : -1 : 1
% if ek == Ng
% Rhs_i = 0*kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% coe=kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1)+1/dt+sigma_c(Ng,1)/2+3/2/dg*S_s(Ng,1)+1/2/dg*(S_s(Ng+1,1)-S_s(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% else
% Rhs_i = -kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)/2)/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:))+kron((3/2/dg*S_s(ek+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% coe=kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(ek,1)+1/dt+sigma_c(ek,1)/2+3/2/dg*S_s(ek,1)+1/2/dg*(S_s(ek+1,1)-S_s(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(Ng,1)/2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)/2)/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% end
% 
% end
Rhs_1e=f_F/dt-kron(sigma_c,ones((Ny+1)*(Nz+1),Nmu*Nom))/2.*f_F;
Rhs_2e(1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron(3/2/dg*S_s(2:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
    +kron(3/2/dg*S_s(1:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F(1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    -kron(3/2/dg*(S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron(1/2/dg*(-S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F(1:(Ny+1)*(Nz+1)*(Ng-1),:));
Rhs_2e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=kron(3/2/dg*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
    -kron(3/2/dg*(S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron(1/2/dg*(-S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
Rhs_3e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
    -kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2.*sigma_c(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    +kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
    -kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
Rhs_3e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
    -kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F(1:(Ny+1)*(Nz+1)*(2-1),:))-kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2.*sigma_c(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:))...
    +kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
    -kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:));
Rhs_3e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2.*sigma_c(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
    +kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    -kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));


Rhs_4e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
    -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2.*sigma_c(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    +kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
    -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
Rhs_4e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
    -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F(1:(Ny+1)*(Nz+1)*(2-1),:))-kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2*sigma_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:))...
    +kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
    -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F(1:(Ny+1)*(Nz+1)*(2-1),:));
Rhs_4e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2*sigma_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
    +kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    -kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
Rhs_e = Rhs_1e+Rhs_2e-Rhs_3e;
for ek = Ng : -1 : 1
if ek == Ng
Rhs_i = 0*kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
coe=kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1)+1/dt+sigma_c(Ng,1)/2+3/2/dg*S_s(Ng,1)+1/2/dg*(S_s(Ng+1,1)-S_s(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
else
Rhs_i = -kron((3/2/dg*S_s(ek+1,1)+0/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)/2)/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:))+kron((3/2/dg*S_s(ek+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
coe=kron((3/2/dg*S_s(ek+1,1)+0/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(ek,1)+1/dt+sigma_c(ek,1)/2+3/2/dg*S_s(ek,1)+1/2/dg*(S_s(ek+1,1)-S_s(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(Ng,1)/2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)/2)/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
end

end
for ek = 1 : Ng
temp_1=F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
temp_2=f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
if all (temp_1(:)==0)&&all (temp_2(:)==0)
continue
else 
temp_1=full(temp_1);temp_2=full(temp_2);
temp_1=reshape(temp_1,Ny+1,Nz+1,[]);temp_2=reshape(temp_2,Ny+1,Nz+1,[]);
temp_3=kron(Omend(:,1)/2,sqrt(-1)*ytemp);temp_4=kron(Omend(:,2)/2,sqrt(-1)*ztemp);
temp_5=(2/dt*ones((Ny+1)*Nmu*Nom,Nz+1)-temp_3-temp_4)./(2/dt*ones((Ny+1)*Nmu*Nom,Nz+1)+temp_3+temp_4);
temp_5=permute(reshape(temp_5',Ny+1,Nz+1,[]),[2 1 3]);
for ord=1:Nmu*Nom
temp_1(:,:,ord)=fft2(temp_1(:,:,ord));
temp_2(:,:,ord)=fft2(temp_2(:,:,ord));
end
temp_1=temp_5.*temp_1;temp_2=temp_5.*temp_2;
for ord=1:Nmu*Nom
temp_1(:,:,ord)=abs(ifft2(temp_1(:,:,ord)));
temp_2(:,:,ord)=abs(ifft2(temp_2(:,:,ord)));
end
temp_1=reshape(temp_1,[],Nmu*Nom);temp_2=reshape(temp_2,[],Nmu*Nom);
% temp_1(temp_1<10^(-10))=0;temp_2(temp_2<10^(-10))=0;
temp_1=sparse(temp_1);temp_2=sparse(temp_2);
F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_1;
f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_2;
end
end
for ek = 1 : Ng
temp_1=F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
temp_2=f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
if all (temp_1(:)==0)&&all (temp_2(:)==0)
continue
else 
temp_1=full(temp_1);temp_2=full(temp_2);
temp_1=reshape(temp_1',Nom,Nmu,[]);temp_2=reshape(temp_2',Nom,Nmu,[]);
temp_6=kron(ones((Ny+1)*(Nz+1),1)/du/du/4*sig_trg(ek),utemp);temp_7=kron(ones((Ny+1)*(Nz+1),1)/dv/dv/4*sig_trg(ek),vtemp);

temp_8=(2/dt*ones((Ny+1)*(Nz+1)*Nom,Nmu)+temp_6+temp_7)./(2/dt*ones((Ny+1)*(Nz+1)*Nom,Nmu)-temp_6-temp_7);
temp_8=permute(reshape(temp_8',Nmu,Nom,[]),[2 1 3]);
for or=1:(Ny+1)*(Nz+1)
temp_1(:,:,or)=fft2(temp_1(:,:,or));
temp_2(:,:,or)=fft2(temp_2(:,:,or));
end
temp_1=temp_8.*temp_1;temp_2=temp_8.*temp_2;
for or=1:(Ny+1)*(Nz+1)
temp_1(:,:,or)=abs(ifft2(temp_1(:,:,or)));
temp_2(:,:,or)=abs(ifft2(temp_2(:,:,or)));
end
temp_1=reshape(temp_1,[],(Ny+1)*(Nz+1));temp_2=reshape(temp_2,[],(Ny+1)*(Nz+1));
% temp_1(temp_1<10^(-10))=0;temp_2(temp_2<10^(-10))=0;
temp_1=sparse(temp_1');temp_2=sparse(temp_2');
F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_1;
f_F((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_2;
end
end
%% calculate secondary proton
for i = 1 : (Ny+1)*(Nz+1)
F_temp ((i-1)*Ng+1:i*Ng,:)=F(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:);
f_Ftemp ((i-1)*Ng+1:i*Ng,:)=f_F(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:);
end
F_temp=reshape(F_temp',(Ny+1)*(Nz+1),[]);f_Ftemp=reshape(f_Ftemp',(Ny+1)*(Nz+1),[]);
for ek = 1 : Ng

F_tempp(:,(ek-1)*(Nom)*(Nmu)+1:(ek)*(Nom)*(Nmu))=F_temp*(kron((ker_e1(ek,:))',ones(Nmu*Nom,Nmu*Nom)).*ker_v);
f_Ftempp(:,(ek-1)*(Nom)*(Nmu)+1:(ek)*(Nom)*(Nmu))=f_Ftemp*(kron((ker_e2(ek,:))',ones(Nmu*Nom,Nmu*Nom)).*ker_v);
end
F_temp=F_tempp;f_Ftemp=f_Ftempp;
% F_temp=F_temp*(kron(ker_e1',ones(Nmu*Nom,Nmu*Nom)).*kron(ker_v,ones(1,Ng)));f_Ftemp=f_Ftemp*(kron(ker_e2',ones(Nmu*Nom,Nmu*Nom)).*kron(ker_v,ones(1,Ng)));
F_temp=reshape(F_temp',[],(Nmu)*(Nom));f_Ftemp=reshape(f_Ftemp',[],(Nmu)*(Nom));
F_temp1=F_temp*dg;f_Ftemp1=f_Ftemp*dg/3;
for i = 1 : (Ny+1)*(Nz+1)
F_temps(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:)=F_temp1((i-1)*Ng+1:i*Ng,:);
f_Ftemps(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:)=f_Ftemp1((i-1)*Ng+1:i*Ng,:);
end
for ek = 1 : Ng
temp_1=F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
temp_2=f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
if all (temp_1(:)==0)&&all (temp_2(:)==0)
continue
else 
temp_1=full(temp_1);temp_2=full(temp_2);
temp_1=reshape(temp_1',Nom,Nmu,[]);temp_2=reshape(temp_2',Nom,Nmu,[]);
temp_6=kron(ones((Ny+1)*(Nz+1),1)/du/du/4*sig_trg(ek),utemp);temp_7=kron(ones((Ny+1)*(Nz+1),1)/dv/dv/4*sig_trg(ek),vtemp);

temp_8=(2/dt*ones((Ny+1)*(Nz+1)*Nom,Nmu)+temp_6+temp_7)./(2/dt*ones((Ny+1)*(Nz+1)*Nom,Nmu)-temp_6-temp_7);
temp_8=permute(reshape(temp_8',Nmu,Nom,[]),[2 1 3]);
for or=1:(Ny+1)*(Nz+1)
temp_1(:,:,or)=fft2(temp_1(:,:,or));
temp_2(:,:,or)=fft2(temp_2(:,:,or));
end
temp_1=temp_8.*temp_1;temp_2=temp_8.*temp_2;
for or=1:(Ny+1)*(Nz+1)
temp_1(:,:,or)=abs(ifft2(temp_1(:,:,or)));
temp_2(:,:,or)=abs(ifft2(temp_2(:,:,or)));
end
temp_1=reshape(temp_1,[],(Ny+1)*(Nz+1));temp_2=reshape(temp_2,[],(Ny+1)*(Nz+1));
% temp_1(temp_1<10^(-10))=0;temp_2(temp_2<10^(-10))=0;
temp_1=sparse(temp_1');temp_2=sparse(temp_2');
F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_1;
f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_2;
end
end

for ek = 1 : Ng
temp_1=F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
temp_2=f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
if all (temp_1(:)==0)&&all (temp_2(:)==0)
continue
else 
temp_1=full(temp_1);temp_2=full(temp_2);
temp_1=reshape(temp_1,Ny+1,Nz+1,[]);temp_2=reshape(temp_2,Ny+1,Nz+1,[]);
temp_3=kron(Omend(:,1)/2,sqrt(-1)*ytemp);temp_4=kron(Omend(:,2)/2,sqrt(-1)*ztemp);
temp_5=(2/dt*ones((Ny+1)*Nmu*Nom,Nz+1)-temp_3-temp_4)./(2/dt*ones((Ny+1)*Nmu*Nom,Nz+1)+temp_3+temp_4);
temp_5=permute(reshape(temp_5',Ny+1,Nz+1,[]),[2 1 3]);
for ord=1:Nmu*Nom
temp_1(:,:,ord)=fft2(temp_1(:,:,ord));
temp_2(:,:,ord)=fft2(temp_2(:,:,ord));
end
temp_1=temp_5.*temp_1;temp_2=temp_5.*temp_2;
for ord=1:Nmu*Nom
temp_1(:,:,ord)=abs(ifft2(temp_1(:,:,ord)));
temp_2(:,:,ord)=abs(ifft2(temp_2(:,:,ord)));
end
temp_1=reshape(temp_1,[],Nmu*Nom);temp_2=reshape(temp_2,[],Nmu*Nom);
% temp_1(temp_1<10^(-10))=0;temp_2(temp_2<10^(-10))=0;
temp_1=sparse(temp_1);temp_2=sparse(temp_2);
F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_1;
f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_2;
end
end
% Rhs_1e=f_F1/dt-kron(sigma_c,ones((Ny+1)*(Nz+1),Nmu*Nom))/2.*f_F1;
% Rhs_2e(1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron(3/2/dg*S_s(2:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     +kron(3/2/dg*S_s(1:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron(3/2/dg*(S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron(1/2/dg*(-S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F1(1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_2e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=kron(3/2/dg*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron(3/2/dg*(S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron(1/2/dg*(-S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs_3e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2.*sigma_c(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     +kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_3e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(2-1),:))-kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2.*sigma_c(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))...
%     +kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:));
% Rhs_3e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2.*sigma_c(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     +kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% 
% Rhs_5e=kron((3/2/dg*S_s(2:Ng+1,1)+3/dg*S_s(1:Ng,1))./(1/dt*ones(Ng,1)+1/2/dg*S_s(1:Ng,1)+sigma_c(1:Ng,1)/2)/2/dg.*S_s(2:Ng+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F_tempf/2+f_Ftempf/2+F_temps/2+f_Ftemps/2);
% Rhs_6e=kron((3/2/dg*S_s(2:Ng+1,1)+3/dg*S_s(1:Ng,1))./(3/2/dg*S_s(2:Ng+1,1)+3/dg*S_s(1:Ng,1))./(1/dt*ones(Ng,1)+1/2/dg*S_s(1:Ng,1)+sigma_c(1:Ng,1)/2)/2/dg.*S_s(2:Ng+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F_tempf/2+f_Ftempf/2+F_temps/2+f_Ftemps/2);
% Rhs_4e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2.*sigma_c(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     +kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_4e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(2-1),:))-kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2*sigma_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))...
%     +kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:));
% Rhs_4e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2*sigma_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     +kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs_e = Rhs_1e+Rhs_2e-Rhs_3e-Rhs_5e;
% for ek = Ng : -1 : 1
% if ek == Ng
% Rhs_i = 0*kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1))/dg/2/dg,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% coe=kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1)+1/dt+sigma_c(Ng,1)/2+3/2/dg*S_s(Ng,1)+1/2/dg*(S_s(Ng+1,1)-S_s(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_6e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% else
% Rhs_i = -kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)/2)/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:))+kron((3/2/dg*S_s(ek+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% coe=kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(ek,1)+1/dt+sigma_c(ek,1)/2+3/2/dg*S_s(ek,1)+1/2/dg*(S_s(ek+1,1)-S_s(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(Ng,1)/2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_6e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)/2)/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% 
% 
% 
% 
% % coe=kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1))/2/dg.*S_s(Ng,1)+1/dt-3/2/dg*S_s(Ng,1)+1/2/dg*(S_s(Ng+1,1)-S_s(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% % f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% % F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_6e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% % else
% % Rhs_i = -kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1))/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:))+kron((3/2/dg*S_s(ek+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% % Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% % coe=kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1))/2/dg.*S_s(ek,1)+1/dt-3/2/dg*S_s(ek,1)+1/2/dg*(S_s(ek+1,1)-S_s(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% % f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% % F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_6e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1))/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% end
% 
% end
% F_tempf=F_temps;f_Ftempf=f_Ftemps;
Rhs_1e=f_F1/dt-kron(sigma_c,ones((Ny+1)*(Nz+1),Nmu*Nom))/2.*f_F1;
Rhs_2e(1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron(3/2/dg*S_s(2:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
    +kron(3/2/dg*S_s(1:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    -kron(3/2/dg*(S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron(1/2/dg*(-S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F1(1:(Ny+1)*(Nz+1)*(Ng-1),:));
Rhs_2e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=kron(3/2/dg*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
    -kron(3/2/dg*(S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron(1/2/dg*(-S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
Rhs_3e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
    -kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2.*sigma_c(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    +kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
    -kron((3/2/dg*S_s(3:Ng,1)+0/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
Rhs_3e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
    -kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(2-1),:))-kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2.*sigma_c(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))...
    +kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
    -kron((3/2/dg*S_s(2,1)+0/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1,1)/2)/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:));
Rhs_3e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2.*sigma_c(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
    +kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    -kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));

Rhs_5e=kron((3/2/dg*S_s(2:Ng+1,1)+0/dg*S_s(1:Ng,1))./(1/dt*ones(Ng,1)+1/2/dg*S_s(1:Ng,1)+sigma_c(1:Ng,1)/2)/2/dg.*S_s(2:Ng+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F_tempf/2+f_Ftempf/2+F_temps/2+f_Ftemps/2);
Rhs_6e=kron((3/2/dg*S_s(2:Ng+1,1)+3/dg*S_s(1:Ng,1))./(3/2/dg*S_s(2:Ng+1,1)+3/dg*S_s(1:Ng,1))./(1/dt*ones(Ng,1)+1/2/dg*S_s(1:Ng,1)+sigma_c(1:Ng,1)/2)/2/dg.*S_s(2:Ng+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F_tempf/2+f_Ftempf/2+F_temps/2+f_Ftemps/2);
Rhs_4e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
    -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2.*sigma_c(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    +kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
    -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(1/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1)+sigma_c(2:Ng-1,1)/2)/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
Rhs_4e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
    -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(2-1),:))-kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2*sigma_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))...
    +kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
    -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(1/dt+1/2/dg*S_s(1,1)+sigma_c(1)/2)/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:));
Rhs_4e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2*sigma_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
    +kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dt,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
    -kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
Rhs_e = Rhs_1e+Rhs_2e-Rhs_3e-Rhs_5e;
for ek = Ng : -1 : 1
if ek == Ng
Rhs_i = 0*kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1))/dg/2/dg,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
coe=kron((3/2/dg*S_s(Ng+1,1)+0/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(Ng,1)+1/dt+sigma_c(Ng,1)/2+3/2/dg*S_s(Ng,1)+1/2/dg*(S_s(Ng+1,1)-S_s(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)/2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_6e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
else
Rhs_i = -kron((3/2/dg*S_s(ek+1,1)+0/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)/2)/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:))+kron((3/2/dg*S_s(ek+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
coe=kron((3/2/dg*S_s(ek+1,1)+0/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(Ng,1)/2)/2/dg.*S_s(ek,1)+1/dt+sigma_c(ek,1)/2+3/2/dg*S_s(ek,1)+1/2/dg*(S_s(ek+1,1)-S_s(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(Ng,1)/2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_6e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)/2)/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));




% coe=kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1))/2/dg.*S_s(Ng,1)+1/dt-3/2/dg*S_s(Ng,1)+1/2/dg*(S_s(Ng+1,1)-S_s(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1)+sigma_c(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_6e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% else
% Rhs_i = -kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1))/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:))+kron((3/2/dg*S_s(ek+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% coe=kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1))/2/dg.*S_s(ek,1)+1/dt-3/2/dg*S_s(ek,1)+1/2/dg*(S_s(ek+1,1)-S_s(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_6e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(1/dt+1/2/dg*S_s(ek,1)+sigma_c(ek,1))/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
end

end
F_tempf=F_temps;f_Ftempf=f_Ftemps;


% 
% 
% Rhs_1e=2*f_F1/dt;
% Rhs_2e(1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron(3/2/dg*S_s(2:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron(3/2/dg*S_s(1:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron(3/2/dg*(S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-1),:))-kron(1/2/dg*(-S_s(1:Ng-1,1)+S_s(2:Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F1(1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_2e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron(3/2/dg*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron(3/2/dg*(S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))-kron(1/2/dg*(-S_s(Ng,1)+S_s(Ng+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs_3e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     +kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/dt*2,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_3e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(2-1),:))...
%     +kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/dt*2,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:));
% Rhs_3e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     +kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/dt*2,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% 
% 
% Rhs_4e((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)=kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/2/dg.*S_s(3:Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/2/dg.*S_s(2:Ng-1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     +kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/dt*2,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/2/dg/dg.*T_c(3:Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*2+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/2/dg/dg.*T_c(1:Ng-2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(Ng-2),:))...
%     -kron((3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(3/2/dg*S_s(3:Ng,1)+3/dg*S_s(2:Ng-1,1))./(2/dt*ones(Ng-2,1)+1/2/dg*S_s(2:Ng-1,1))/dg/dg.*T_c(2:Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(Ng-1),:));
% Rhs_4e(1:(Ny+1)*(Nz+1)*(2-1),:)=kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/2/dg.*S_s(2,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:)-f_F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/2/dg.*S_s(1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:)-f_F1(1:(Ny+1)*(Nz+1)*(2-1),:))...
%     +kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/dt*2,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:))+kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/2/dg/dg.*T_c(2),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)+1:(Ny+1)*(Nz+1)*(2),:))...
%     -kron((3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(3/2/dg*S_s(2,1)+3/dg*S_s(1,1))./(2/dt+1/2/dg*S_s(1,1))/dg/2/dg.*T_c(1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1(1:(Ny+1)*(Nz+1)*(2-1),:));
% Rhs_4e((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)=-kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/2/dg.*S_s(Ng,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:)-f_F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))...
%     +kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/dt*2,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:))+kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/2/dg/dg.*T_c(Ng-1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-2)+1:(Ny+1)*(Nz+1)*(Ng-1),:))...
%     -kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/dg/2/dg.*T_c(Ng),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs_e = Rhs_1e+Rhs_2e-Rhs_3e;
% for ek = Ng : -1 : 1
% if ek == Ng
% Rhs_i = 0*kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(1/dt+1/2/dg*S_s(Ng,1))/dg/2/dg,ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F((Ny+1)*(Nz+1)*(Ng-1)+1:(Ny+1)*(Nz+1)*(Ng),:));
% Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% coe=kron((3/2/dg*S_s(Ng+1,1)+3/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1))/2/dg.*S_s(Ng,1)+2/dt-3/2/dg*S_s(Ng,1)+1/2/dg*(S_s(Ng+1,1)-S_s(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(Ng,1))./(2/dt+1/2/dg*S_s(Ng,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% else
% Rhs_i = -kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(2/dt+1/2/dg*S_s(ek,1))/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:))+kron((3/2/dg*S_s(ek+1,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% Rhs = Rhs_i + Rhs_e ((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
% coe=kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(2/dt+1/2/dg*S_s(ek,1))/2/dg.*S_s(ek,1)+2/dt-3/2/dg*S_s(ek,1)+1/2/dg*(S_s(ek+1,1)-S_s(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom));
% f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=Rhs./coe;
% F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=kron((1/2/dg*S_s(ek,1))./(2/dt+1/2/dg*S_s(ek,1)),ones((Ny+1)*(Nz+1),Nmu*Nom)).*f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+Rhs_4e((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)+kron((3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(3/2/dg*S_s(ek+1,1)+3/dg*S_s(ek,1))./(2/dt+1/2/dg*S_s(ek,1))/dg/2*S_s(ek+1,1),ones((Ny+1)*(Nz+1),Nmu*Nom)).*(F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:)-f_F1((Ny+1)*(Nz+1)*(ek)+1:(Ny+1)*(Nz+1)*(ek+1),:));
% end
% 
% end

for ek = 1 : Ng
temp_1=F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
temp_2=f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
if all (temp_1(:)==0)&&all (temp_2(:)==0)
continue
else 
temp_1=full(temp_1);temp_2=full(temp_2);
temp_1=reshape(temp_1,Ny+1,Nz+1,[]);temp_2=reshape(temp_2,Ny+1,Nz+1,[]);
temp_3=kron(Omend(:,1)/2,sqrt(-1)*ytemp);temp_4=kron(Omend(:,2)/2,sqrt(-1)*ztemp);
temp_5=(2/dt*ones((Ny+1)*Nmu*Nom,Nz+1)-temp_3-temp_4)./(2/dt*ones((Ny+1)*Nmu*Nom,Nz+1)+temp_3+temp_4);
temp_5=permute(reshape(temp_5',Ny+1,Nz+1,[]),[2 1 3]);
for ord=1:Nmu*Nom
temp_1(:,:,ord)=fft2(temp_1(:,:,ord));
temp_2(:,:,ord)=fft2(temp_2(:,:,ord));
end
temp_1=temp_5.*temp_1;temp_2=temp_5.*temp_2;
for ord=1:Nmu*Nom
temp_1(:,:,ord)=abs(ifft2(temp_1(:,:,ord)));
temp_2(:,:,ord)=abs(ifft2(temp_2(:,:,ord)));
end
temp_1=reshape(temp_1,[],Nmu*Nom);temp_2=reshape(temp_2,[],Nmu*Nom);
% temp_1(temp_1<10^(-10))=0;temp_2(temp_2<10^(-10))=0;
temp_1=sparse(temp_1);temp_2=sparse(temp_2);
F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_1;
f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_2;
end
end
for ek = 1 : Ng
temp_1=F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
temp_2=f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:);
if all (temp_1(:)==0)&&all (temp_2(:)==0)
continue
else 
temp_1=full(temp_1);temp_2=full(temp_2);
temp_1=reshape(temp_1',Nom,Nmu,[]);temp_2=reshape(temp_2',Nom,Nmu,[]);
temp_6=kron(ones((Ny+1)*(Nz+1),1)/du/du/4*sig_trg(ek),utemp);temp_7=kron(ones((Ny+1)*(Nz+1),1)/dv/dv/4*sig_trg(ek),vtemp);

temp_8=(2/dt*ones((Ny+1)*(Nz+1)*Nom,Nmu)+temp_6+temp_7)./(2/dt*ones((Ny+1)*(Nz+1)*Nom,Nmu)-temp_6-temp_7);
temp_8=permute(reshape(temp_8',Nmu,Nom,[]),[2 1 3]);
for or=1:(Ny+1)*(Nz+1)
temp_1(:,:,or)=fft2(temp_1(:,:,or));
temp_2(:,:,or)=fft2(temp_2(:,:,or));
end
temp_1=temp_8.*temp_1;temp_2=temp_8.*temp_2;
for or=1:(Ny+1)*(Nz+1)
temp_1(:,:,or)=abs(ifft2(temp_1(:,:,or)));
temp_2(:,:,or)=abs(ifft2(temp_2(:,:,or)));
end
temp_1=reshape(temp_1,[],(Ny+1)*(Nz+1));temp_2=reshape(temp_2,[],(Ny+1)*(Nz+1));
% temp_1(temp_1<10^(-10))=0;temp_2(temp_2<10^(-10))=0;
temp_1=sparse(temp_1');temp_2=sparse(temp_2');
F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_1;
f_F1((Ny+1)*(Nz+1)*(ek-1)+1:(Ny+1)*(Nz+1)*(ek),:)=temp_2;
end
end
for i = 1 : (Ny+1)*(Nz+1)
F_temp ((i-1)*Ng+1:i*Ng,:)=F_tot(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:);
f_Ftemp ((i-1)*Ng+1:i*Ng,:)=f_Ftot(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:);
end
F_temp=reshape(F_temp',(Ny+1)*(Nz+1),[]);f_Ftemp=reshape(f_Ftemp',(Ny+1)*(Nz+1),[]);
for ek = 1 : Ng

F_tempp(:,(ek-1)*(Nom)*(Nmu)+1:(ek)*(Nom)*(Nmu))=F_temp*(kron((ker_e1(ek,:))',ones(Nmu*Nom,Nmu*Nom)).*ker_v);
f_Ftempp(:,(ek-1)*(Nom)*(Nmu)+1:(ek)*(Nom)*(Nmu))=f_Ftemp*(kron((ker_e2(ek,:))',ones(Nmu*Nom,Nmu*Nom)).*ker_v);
end
F_temp=F_tempp;f_Ftemp=f_Ftempp;
% F_temp=F_temp*(kron(ker_e1',ones(Nmu*Nom,Nmu*Nom)).*kron(ker_v,ones(1,Ng)));f_Ftemp=f_Ftemp*(kron(ker_e2',ones(Nmu*Nom,Nmu*Nom)).*kron(ker_v,ones(1,Ng)));
F_temp=reshape(F_temp',[],(Nmu)*(Nom));f_Ftemp=reshape(f_Ftemp',[],(Nmu)*(Nom));
F_temp1=F_temp*dg;f_Ftemp1=f_Ftemp/3*dg;
for i = 1 : (Ny+1)*(Nz+1)
F_tempt(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:)=F_temp1((i-1)*Ng+1:i*Ng,:);
f_Ftempt(i:(Ny+1)*(Nz+1):(Ny+1)*(Nz+1)*(Ng),:)=f_Ftemp1((i-1)*Ng+1:i*Ng,:);
end
F_tot = F+F1; f_Ftot=f_F+f_F1;
dose_1=sum(sum(F_tot.*kron(sparse(S_s(1:Ng)*dg/2+S_s(2:Ng+1)*dg/2+sigma_c(1:Ng)*dg.*en(2:Ng+1)-sigma_c(1:Ng)*dg*dg/2),ones((Ny+1)*(Nz+1),Nmu*Nom))+f_Ftot.*kron(sparse(-S_s(1:Ng)*dg/6+S_s(2:Ng+1)*dg/6+sigma_c(1:Ng)/6*dg*dg),ones((Ny+1)*(Nz+1),Nmu*Nom))))*dt*du*dv*dy*dz;
dose_2=sum(sum(F_tot(1:(Ny+1)*(Nz+1),:)*S_s(1)*en(1)-f_Ftot(1:(Ny+1)*(Nz+1),:)*S_s(1)*en(1)))*dt*du*dv*dy*dz;
 dose_3=sum(sum((F_tempt+f_Ftempt).*kron(en(2:Ng+1).^2/2-en(1:Ng).^2/2,ones((Ny+1)*(Nz+1),Nmu*Nom))))*dt*du*dv*dy*dz;
fin(kk1+1)=dose_1+dose_2-dose_3;
% QQ=zeros(Ny+1,Nz+1);qq_1=zeros(Ny+1,Nz+1,Nmu*Nom);
% for ek = 1 : Ng
% for j = 1 : Nmu*Nom
% 
% %     Uu= u /sqrt(sig_trg(ek));  Vv= v/sqrt(sig_trg(ek));
% % for jj = 1 : Nmu
% %     Ome((jj-1)*Nom+1:(jj)*Nom,2)=Vv(Nmu-jj+1);
% %     Ome((jj-1)*Nom+1:(jj)*Nom,1)=Uu(1:Nom);
% % end
% %     qq_1(:,:,j) = qq_1(:,:,j)+2/Nmu*2/Nmu*f_1(:,:,j)*(f_2(ek)*dg/2*(S_s(ek)+S_s(ek+1))+f_3(ek)*dg/3/2*(-S_s(ek)+S_s(ek+1)));
% %     if ek == 1
% %     qq_1(:,:,j) = qq_1(:,:,j)+2/Nmu*2/Nmu*f_1(:,:,j)*(f_2(ek)-f_3(ek))*(S_s(ek));
% %     end
%     qq_1(:,:,j) = qq_1(:,:,j)+2/Nmu*2/Nmu*f_1(:,:,j)*(f_2(ek)*dg/2*(S_s(ek)+S_s(ek+1)));
%     if ek == 1
%     qq_1(:,:,j) = qq_1(:,:,j)+2/Nmu*2/Nmu*f_1(:,:,j)*(f_2(ek))*(S_s(ek));
%     end
% end
%     
% end
% 
% for  k = 1: Nmu*Nom
% QQ=QQ+qq_1(:,:,k)*dt;    
% end
% for s = 1 : Nz*Ny
% fin(kk+1)=fin(kk+1)+4/Ny*4/Nz*QQ(s);
% end
% QQ=abs(QQ);
% F=F_fi;


    t = t + dt
    kk1 = kk1 + 1;
%     Re(:,:,kk)=QQ;
end



   if abs(t-0.1*clock)<dt  
      clock = clock+1;
   end
   save('C:\Users\Administrator\Desktop\proton\2.mat','fin')
%   filename = ['APtransport_May31_Nx_',num2str(Nx),'_eps_1e'];
%   save(filename);
  
  
  
  
  
  
  
  
  
  
  
  
  return
function y=cross(Zt,Ne,En)
m_e=9.10956*10^(-31);
m_p=1.673*10^(-27);
alpha=1/137;
v=sqrt(2*En*1.6*10^(-13)/(1.67*10^(-27)));
beta = v/3/10^8;
eta=Zt^(2/3)*alpha^2*m_e*m_e/m_p/m_p./beta./beta;
mid=log((eta+ones(Ne,1))./eta)-ones(Ne,1)./(eta+ones(Ne,1));
sigma_tr=2*3.14*3.34*197*197/137/137/4*Zt^2./En.^2.*mid/10^3;
y=sigma_tr;
return 
function y=stopping(En)
v=sqrt(2*En*1.6*10^(-13)/(1.67*10^(-27))); %m s-1
I =78;%eV
% beta = v/2.998/10^8;
beta_2 = En.*(En+2*938.3*En./En)./(En+938.3*En./En).^2;
F_beta = log(1.02*10^6*beta_2./(1-beta_2))-beta_2-4.31;
y=0.170./beta_2.*F_beta+0.02;
% y=1.0013*y;
return 
function y=cross_1(Zt,En)
m_e=9.10956*10^(-31);
m_p=1.673*10^(-27);
alpha=1/137;
v=sqrt(2*En*1.6*10^(-13)/(1.67*10^(-27)));
beta = v/3/10^8;
eta=Zt^(2/3)*alpha^2*m_e*m_e/m_p/m_p./beta./beta;
mid=log((eta+1)./eta)-1/(eta+1);
sigma_tr=2*3.14*3.34*197*197/137/137/4*Zt^2./En.^2.*mid/10^3;
y=sigma_tr;
return 
function y=stra(Ne,En)
% m_e=9.10956*10^(-31);
% m_p=1.673*10^(-27);
% alpha=1/137;
v=sqrt(2*En*1.6*10^(-13)/(1.67*10^(-27)));
beta = v/3/10^8;
eta=8.99^2*10^18*4*pi*(1.6*10^(-19))^4/(1.6*10^(-13))^2/100;

sigma_tr1=eta*3.34*10^(29)*2/10*(ones(Ne,1)+4/3*19./(1/2*1.02*10^6*beta.^2).*log(1/19*1.02*10^6*beta.^2))+eta*3.34*10^(29)*8/10*(ones(Ne,1)+4/3*105./(1/2*1.02*10^6*beta.^2).*log(1/105*1.02*10^6*beta.^2));
y=sigma_tr1*1.2;
return
