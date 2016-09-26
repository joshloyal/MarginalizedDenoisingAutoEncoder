train_file = 'kin8nm_train.csv';

train = csvread(train_file);
X = train(:,1:end-1);

ncol = size(X, 2);

[h, W] = mDA(X', 0.5, 1e-5);

csvwrite('hidden_state.csv', h)
csvwrite('weights.csv', W)
