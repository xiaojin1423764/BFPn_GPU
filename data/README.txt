# data/README.txt
BFPn Solver Data Files
======================

This directory should contain the following data files:

1. data_cross.txt
   Format: 3 columns per line (mu1, mu2, mu3)
   Lines: Ng (number of energy groups)
   Description: Cross section parameters for each energy group

2. cross_total.txt
   Format: 1 column per line (sigma_c)
   Lines: Ng
   Description: Total cross section for each energy group

Example data generation:
-----------------------
For testing purposes, you can generate synthetic data using the provided
Python script (generate_test_data.py) or use the following MATLAB code:

% Generate test data
Ng = 500;
mu = zeros(Ng, 3);
sigma_c = zeros(Ng, 1);

for i = 1:Ng
    E = (i-0.5)*0.518 + 1;  % Energy in MeV
    % Fill in appropriate physics models
    mu(i,:) = [...];  % Your model here
    sigma_c(i) = [...];
end

save data_cross.txt mu -ascii;
save cross_total.txt sigma_c -ascii;

Note:
-----
The original MATLAB code references:
C:\Users\Administrator\Desktop\proton\data_cross.txt
C:\Users\Administrator\Desktop\proton\cross_total.txt

Please copy these files to this directory before running the solver.
