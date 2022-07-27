clc; 
close all;
clear all;

labels = {'white', 'red'};
labels_values = [1, 2];

tabla_vinos = readtable('winequalityN.csv'); 
[was_found, index] = ismember(tabla_vinos.type, labels); 
l_values = nan(length(index), 1);
l_values(was_found) = labels_values(index(was_found));
tabla_vinos.type = l_values;
wines_matrix = tabla_vinos{:,:};

clear tabla_vinos  l_values  labels_values  labels  index  was_found;

X = wines_matrix(:,2:end);
X(isnan(X))=0.000001;
y = wines_matrix(:,1);

[m, n] = size(X);

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);

[betas, cost, i] = fmincg(@(t)(costFunction(t, X, y)),initial_theta);

m = size(X, 1); 
p = round(sigmoid(X * betas));

error = 1/m*(sum(abs(y-p)));

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Error de entrenamiento: %f\n', error * 100);









