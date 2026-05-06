# %%
from dataclasses import dataclass
import pandas as pd
from enum import IntEnum
import random


class Quality(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class Product:
    price: int
    quality: Quality


@dataclass
class User:
    budget: float
    quality: Quality

    def matches_budget(self, price: int) -> bool:
        return price <= self.budget

    def matches_quality(self, quality: Quality) -> bool:
        return quality >= self.quality

    def is_willing_to_buy(self, price: int, quality: Quality) -> bool:
        return self.matches_budget(price) and self.matches_quality(quality)

    def choose(self, products: list[Product]) -> Product | None:
        for product in sorted(products, key=lambda p: (p.price, p.quality)):
            if self.is_willing_to_buy(product.price, product.quality):
                return product


def create_user(budget: int, std: int | None, quality: Quality):
    if std is not None:
        budget = int(random.gauss(budget, std))
    return User(budget=budget, quality=quality)


def batch_create_users(n: int, budget: int, std: int, quality: Quality):
    return [create_user(budget, std, quality) for _ in range(n)]


@dataclass
class Transaction:
    user: User
    product: Product
    period: int


pro_users = batch_create_users(n=100, budget=500, std=80, quality=Quality.HIGH)
avg_users = batch_create_users(n=250, budget=300, std=50, quality=Quality.MEDIUM)
low_users = batch_create_users(n=350, budget=200, std=20, quality=Quality.LOW)

users = pro_users + avg_users + low_users


def batch_create_products(n: int, price: int, std: int | None, quality: Quality):
    if std:
        price = int(random.gauss(price, std))
    return [Product(price=price, quality=quality) for _ in range(n)]


products = [
    *batch_create_products(n=10, price=200, std=20, quality=Quality.LOW),
    *batch_create_products(n=15, price=350, std=30, quality=Quality.MEDIUM),
    *batch_create_products(n=10, price=400, std=40, quality=Quality.HIGH),
]


class Simulation:
    def __init__(self, users, products):
        self.period = 0
        self.products = products
        self.users = users
        self.transactions = []

    def advance(self):
        self.period += 1
        for user in self.users:
            product = user.choose(self.products)
            if product:
                self.transactions.append(
                    Transaction(user=user, product=product, period=self.period)
                )

    def to_df(self):
        return pd.DataFrame(
            [
                {
                    "period": t.period,
                    "user_budget": t.user.budget,
                    "user_quality": t.user.quality.name,
                    "product_price": t.product.price,
                    "product_quality": t.product.quality.name,
                }
                for t in self.transactions
            ]
        )


sim = Simulation(users=users, products=products)

periods = list(range(3))
for period in periods:
    sim.advance()

df = sim.to_df()

# %%
import altair as alt

alt.Chart(df).mark_bar().encode(
    x=alt.X("user_budget:Q").bin(maxbins=30),
    y="count()",
    color="product_quality:N",
    tooltip=["count()", "product_quality:N"],
)
