function elbow_ix = findElbow(arr)

med_arr = medfilt1([arr(1); arr; arr(end)]);
med_arr = med_arr(2:end-1);

elbow_ix = find(med_arr == max(med_arr), 1);

end