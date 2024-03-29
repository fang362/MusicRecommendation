# Thanks to Siraj Raval for this module
# Refer to https://github.com/llSourcell/recommender_live for more details

import numpy as np
import pandas

# 基于排行榜榜单推荐系统 类
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None

    # create方法
    # 基于给定的训练数据、用户ID和歌曲ID来 创建推荐系统模型。
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # 计算每首歌被多少用户听过
        # 使用groupby方法按物品ID对数据进行分组
        # agg方法用于聚合数据，这里使用count函数来计算每个组的用户数量之和
        # 对每个组中的 self.user_id 列使用 count 函数，以计算不同歌曲被播放的次数
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        # 将user_id列 重命名为`score`，表示流行度得分。
        train_data_grouped.rename(columns = {user_id: 'score'},inplace=True)
    
        # 根据歌曲的被播放总次数 排序（降序）
        # 如果被播放次数相同，按用户id（升序）。
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        # 基于得分产生一个推荐排名（降序）
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #选择推荐排名中的前十首歌曲作为推荐结果，存入self.popularity_recommendations属性中
        # popularity_recommendations为df数据类型
        self.popularity_recommendations = train_data_sort.head(10)

    # Recommend方法 推荐操作
    def recommend(self, user_id):
        # 获取传入的用户id 对应的popularity_recommendations属性值
        user_recommendations = self.popularity_recommendations
        
        # 给popularity_recommendations添加一个新的列，列名为user_id，赋值为 传入的用户id
        user_recommendations['user_id'] = user_id
    
        # 调整列的顺序，将user_id列移至最前面
        # 将 user_recommendations DataFrame 的列名作为元素存入列表
        cols = user_recommendations.columns.tolist()
        # cols[-1:]取出列表的最后一个元素，cols[:-1]取出列表除最后一个元素的所有元素，拼接操作。
        cols = cols[-1:] + cols[:-1]
        # 此df数据user_recommendations 的列名顺序为cols列表中的列名顺序
        user_recommendations = user_recommendations[cols]

        # 返回推荐信息
        return user_recommendations
    

# 基于物品的协同过滤推荐系统
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given user
    # 得到给定用户的听过的所有歌(重复的歌曲不计入)
    def get_user_items(self, user):
        # 拿到当前用户听过的所有歌（一首歌可能出现很多次）
        # self.train_data[...] 使用布尔条件 过滤 train_data
        # 得到一个只包含指定用户的听歌信息
        user_data = self.train_data[self.train_data[self.user_id] == user]
        # 提取歌曲ID列名去重+变成列表
        user_items = list(user_data[self.item_id].unique())

        return user_items
        
    #Get unique users for a given item (song)
    # 得到给定的歌曲被哪些人听过
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #Get unique items (songs) in the training data
    # 得到去重的歌曲列表
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items


    #构建共现矩阵
    # 目标：遍历得出 66*4879矩阵中每首歌之间（用户听过的与歌曲库之间的歌）的相似度
    def construct_cooccurence_matrix(self, user_songs, all_songs):

        user_songs_users = []
        # 拿到用户听过的歌
        for i in range(0, len(user_songs)):
            # 找出这些歌还被哪些用户听过
            # 将这些新找到的用户存入 "与推荐用户听过同一首歌的用户" 的列表中
            user_songs_users.append(self.get_item_users(user_songs[i]))

        #初始化矩阵 len(user_songs) X len(songs) ——> 66*4879
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        # 接下来的目标：找出此用户听过的每一首歌 与 歌曲库的每一首歌 之间的关系
        # 如何确定关系？
        # 假设此用户听过 歌曲i，从歌曲库中选中 歌曲j；
        # i被8000人听过，j被7000人听过
        # 如果听过这首歌的这两个群体 大部分是同一个人，那么歌曲i与j的相似度很高
        # 公式化表示：( User_i 交 User_j )/( User_i 并 User_j )
        # 值越大，歌曲i与j受众越相似，说明i与j相似度越高

        # 遍历歌曲库的所有歌4879首
        for i in range(0,len(all_songs)):
            # 找到歌库中每首歌的所有信息
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            # 找到 听过"遍历到的这首歌的"所有user
            users_i = set(songs_i_data[self.user_id].unique())

            # 遍历"推荐用户"听过的歌曲
            for j in range(0,len(user_songs)):
                # 将已经得到的"与推荐用户听过同一首歌的用户" 传入users_j列表中
                users_j = user_songs_users[j]
                # 计算两类人的交集
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                # 术语：计算共现矩阵中位置 [i,j] 的值作为Jaccard指数。
                # 如果用户的交集不等于0
                if len(users_intersection) != 0:
                    #计算两类人的并集
                    users_union = users_i.union(users_j)
                    # 矩阵中每个元素的值为 ：交集/并集
                    # 得出的计算结果就是两首歌的相似度
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0

        return cooccurence_matrix

    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("矩阵中非0值的数量:%d" % np.count_nonzero(cooccurence_matrix))

        # 计算歌曲库中的每首歌(与用户听过的所有歌)的平均相似度！注意是取平均值
        # 假设 r1=j1*i1, 则歌曲j1的平均值：(j1*i1 + j1*i2 +...+ j1*i66)/66 = j1(i1+i2+...+i66)/66
        # 在所有歌曲jn中，j几的平均得分最高，也就是歌曲库中哪首歌的相似度最高，则推荐哪个

        # 如果 cooccurence_matrix 的形状是 (m, n)
        # 那么 cooccurence_matrix.sum(axis=0) 的结果将是一个长度为 n 的一维数组
        # 这个数组的每个元素都是 cooccurence_matrix 中对应 "列的总和"
        # 在上面代码中，cooccurence_matrix.sum(axis=0)的结果是：长度为4879的一维数组
        # float(cooccurence_matrix.shape[0])=66
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
 
        #将这些得分排序(降序）
        sort_index = sorted(( (i,e)for i,e in enumerate( list(user_sim_scores) ) ), reverse=True)
    
        #Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        #将数组中的值作为列名赋值给df
        df = pandas.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        # 用基于歌曲相似度推荐得到的前10个结果 填到df中
        rank = 1 # 排名计数变量
        for i in range(0,len(sort_index)):
            # 每首歌曲满足以下条件 可执行：
            #
            # 得分不是 NaN
            # 该歌曲不在用户已听过的歌曲列表中
            # 尚未达到 10 个推荐的上限
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                # 如果满足上述条件，则在 DataFrame df 中添加一行，包含用户ID、歌曲名称、得分和排名
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                # 添加一条信息，更新一次排名
                rank = rank+1
        
        #如果df的行数为0，则表示没有推荐的歌曲
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    # 为特定用户生成推荐
    def recommend(self, user):

        #A. Get all unique songs for this user
        user_songs = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(user_songs))

        #B. Get all unique items (songs) in the training data
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))

        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        #D. Use the cooccurence matrix to make recommendations
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    #Get similar items to given items
    # 为给定的歌曲列表生成相似的歌曲
    def get_similar_items(self, item_list):
        
        user_songs = item_list

        #B. Get all unique items (songs) in the training data
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))

        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        #D. Use the cooccurence matrix to make recommendations
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations