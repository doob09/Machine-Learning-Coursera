#load data 
data = load('ex2data1.txt');

#get the features and label
X = data(:,[1,2]);
y = data(:,3);

#Plot the data
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);


plot(X,y,'k+','ko','MarkerSize',10);

