function one_encoded = one_hot(num,num_labels)
#input as numbe 0-9 -> oupt : one at index and 0 out others
one_encoded  = zeros(num_labels,1)
#need if check >0 and <= num_labels
one_encoded(num,:) = 1
end