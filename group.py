import random
from collections import defaultdict

class FindGroup:
    def __init__(self, file_path, threshold_min=0.2, threshold_max=0.8, n=3):
        self.file_path = file_path
        self.threshold_min = threshold_min  # 最小概率阈值
        self.threshold_max = threshold_max  # 最大概率阈值
        self.n = n  # 随机选择的商品数量
        self.product_frequency = defaultdict(int) #每个商品出现的次数
        self.user_products = {}  # 记录每个用户交互的商品
        self.total_users = 0
        self.product_probabilities = {}  # 存储每个产品的出现概率
        self._load_data()  # 读取数据

    def _load_data(self):
        """加载数据，计算每个产品的出现频率和概率"""
        with open(self.file_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                user_id = data[0]  # 用户ID
                products = set(data[1:]) 
                self.total_users += 1
                self.user_products[user_id] = products  # 记录用户交互的商品
                for product in products:
                    self.product_frequency[product] += 1

        self.product_probabilities = {
            product: freq / self.total_users for product, freq in self.product_frequency.items()
        }

    def get_filtered_products(self):
        """获取概率介于 threshold_min 和 threshold_max 之间的产品"""
        # return [product for product, prob in self.product_probabilities.items()
        #         if self.threshold_min < prob < self.threshold_max]
        filtered_products = [product for product, prob in self.product_probabilities.items()
                           if self.threshold_min < prob < self.threshold_max]
        filtered_products.sort(key=int)
        return filtered_products

    def select_random_products(self, filtered_products):
        """从筛选出的产品中随机选 self.n 个"""
        if len(filtered_products) < self.n:
            raise ValueError(f"符合概率范围的产品不足 {self.n} 个，无法随机选择！")
        return random.sample(filtered_products, self.n)

    def find_common_users(self, selected_products):
        return {user for user, products in self.user_products.items() if set(selected_products).issubset(products) and len(selected_products)/len(products) > 0}

    def analyze(self):
        filtered_products = self.get_filtered_products()
        if len(filtered_products) < self.n:
            print("********************")
            print(f"符合概率范围的商品不足 {self.n} 个，无法随机选择！")
            print("********************")
            return [], set()

        selected_products = self.select_random_products(filtered_products)
        common_users = self.find_common_users(selected_products)


        interested_items_set = set(map(int, selected_products)) 
        target_user_set = set(map(int, common_users))

        print("********************")
        print(f"随机选出的 {self.n} 个符合概率要求的商品: {interested_items_set}")
        print(f"与 {interested_items_set} 这 {self.n} 个商品都交互过的用户有 {len(target_user_set)} 个，分别是: {target_user_set}")
        print("********************")


        return interested_items_set, target_user_set