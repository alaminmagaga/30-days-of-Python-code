from collections import Counter
import math

# 1
class Statistics:
    def __init__(self, data):
        self.data = data
        
    def count(self):
        return len(self.data)
    
    def sum(self):
        return sum(self.data)
    
    def min(self):
        return min(self.data)
    
    def max(self):
        return max(self.data)
    
    def range(self):
        return max(self.data) - min(self.data)
    
    def mean(self):
        return sum(self.data) / len(self.data)
    
    def median(self):
        n = len(self.data)
        sorted_data = sorted(self.data)
        if n % 2 == 0:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        else:
            return sorted_data[n // 2]
    
    def mode(self):
        count_dict = Counter(self.data)
        mode, count = count_dict.most_common(1)[0]
        return {'mode': mode, 'count': count}
    
    def std(self):
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.data) / (len(self.data) - 1)
        return math.sqrt(variance)
    
    def var(self):
        mean = self.mean()
        return sum((x - mean) ** 2 for x in self.data) / (len(self.data) - 1)
    
    def freq_dist(self):
        freq_dict = Counter(self.data)
        freq_dist = [(freq_dict[k] / len(self.data) * 100, k) for k in freq_dict]
        return sorted(freq_dist, reverse=True)
    
    def describe(self):
        return f"Count: {self.count()}\nSum: {self.sum()}\nMin: {self.min()}\nMax: {self.max()}\nRange: {self.range()}\nMean: {self.mean()}\nMedian: {self.median()}\nMode: {self.mode()}\nVariance: {self.var()}\nStandard Deviation: {self.std()}\nFrequency Distribution: {self.freq_dist()}"
    
ages = [31, 26, 34, 37, 27, 26, 32, 32, 26, 27, 27, 24, 32, 33, 27, 25, 26, 38, 37, 31, 34, 24, 33, 29, 26]
data = Statistics(ages)

print('Count:', data.count()) # 25
print('Sum: ', data.sum()) # 744
print('Min: ', data.min()) # 24
print('Max: ', data.max()) # 38
print('Range: ', data.range()) # 14
print('Mean: ', data.mean()) # 30
print('Median: ', data.median()) # 29
print('Mode: ', data.mode()) # {'mode': 26, 'count': 5}
print('Standard Deviation: ', data.std()) # 4.2
print('Variance: ', data.var()) # 17.5
print('Frequency Distribution: ', data.freq_dist()) # [(20.0, 26), (16.0, 27), (12.0, 32), (8.0, 37), (8.0, 34), (8.0, 33), (8.0, 31), (8.0

# 2
class PersonAccount:
    def __init__(self, firstname, lastname):
        self.firstname = firstname
        self.lastname = lastname
        self.incomes = set()
        self.expenses = set()

    def add_income(self, amount, description):
        self.incomes.add((amount, description))

    def add_expense(self, amount, description):
        self.expenses.add((amount, description))

    def total_income(self):
        return sum(amount for amount, _ in self.incomes)

    def total_expense(self):
        return sum(amount for amount, _ in self.expenses)

    def account_balance(self):
        return self.total_income() - self.total_expense()

    def account_info(self):
        info = f"{self.firstname} {self.lastname}'s account:\n"
        info += f"Total income: {self.total_income()}\n"
        info += f"Total expenses: {self.total_expense()}\n"
        info += f"Account balance: {self.account_balance()}"
        return info
