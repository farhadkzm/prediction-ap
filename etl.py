group_min_size = 1000
group_pick_size = 10
number_of_group = 10
groups = df.groupby('RECEIVER_SUBURB').size()
g1k = groups[groups > group_min_size]
g1k = list(g1k.index)
df1k = df[df['RECEIVER_SUBURB'].isin(g1k)]
working_set = df1k.groupby('RECEIVER_SUBURB').head(group_pick_size).sort_values('RECEIVER_SUBURB').iloc[0:number_of_group * group_pick_size]
