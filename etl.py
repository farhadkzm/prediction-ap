def smart_split(df,groupby_col, number_of_group=10, group_pick_size=1000):

    groups = df.groupby(groupby_col).size()
    g1k = groups[groups > group_pick_size]
    g1k = list(g1k.index)
    df1k = df[df[groupby_col].isin(g1k)]

    total_rows = number_of_group * group_pick_size
    working_set = df1k.groupby(groupby_col).head(group_pick_size).sort_values(groupby_col).iloc[
                  0:total_rows]
    grouped = working_set.groupby(groupby_col)

    return grouped
