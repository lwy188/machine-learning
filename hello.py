#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import requests
from lxml import etree
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt



def get_house_info(urls, output_path):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36'}
    data = []
    for url in urls:
        r = requests.get(url, headers=header)
        if r.status_code == 200:
            html = etree.HTML(r.text)
            all_info = html.xpath('//div[@class="info clear"]')
            for info in all_info:
                # ' | 2室1厅 | 79平米 | 南 | 精装'
                houseInfo = info.xpath('div[@class="address"]/div[@class="houseInfo"]/text()')[0]
                positionInfo = info.xpath('div[@class="flood"]/div[@class="positionInfo"]/text()')[0]
                price = info.xpath('div[@class="priceInfo"]/div[@class="totalPrice"]/span/text()')[0]

                room_info = re.search(r'(\d)室(\d)厅', houseInfo)
                decorate = re.search(r'(精装|毛坯|简装)', houseInfo)
                area = re.search(r'(\d+(?:.\d+)*)平米', houseInfo)
                floor = re.search(r'(低|中|高)楼层', positionInfo)

                if not (room_info and room_info and decorate and area and floor):
                    continue

                rooms = room_info.group(1)
                halls = room_info.group(2)
                decorate = {"毛坯": 0, "简装": 1, "精装": 2}[decorate.group(1)]
                area = area.group(1)
                floor = {"低": 0, "中": 1, "高": 2}[floor.group(1)]

                data.append([rooms, halls, decorate, area, floor, price])

    # 写入csv文件
    with open(output_path, "w", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入列的名称
        writer.writerow(["房间数", "客厅数", "装修(毛坯:0,简装:1,精装:2)", "面积", "楼层(低:0,中:1,高:2)", "总价"])
        # 写入内容
        writer.writerows(data)


sounth_urls = ("https://cd.lianjia.com/ershoufang/c3011056655752/",
               "https://cd.lianjia.com/ershoufang/pg2c3011056655752/",
               "https://cd.lianjia.com/ershoufang/pg3c3011056655752/",
               "https://cd.lianjia.com/ershoufang/pg4c3011056655752/")
north_urls = ("https://cd.lianjia.com/ershoufang/c3011056658392/",
              "https://cd.lianjia.com/ershoufang/pg2c3011056658392/")

get_house_info(sounth_urls, 'C:\project\python\南区.csv')
get_house_info(north_urls, 'C:\project\python\北区.csv')
data = pd.read_csv('C:\project\python\南区.csv')
data_test = pd.read_csv('C:\project\python\北区.csv')
plt.scatter(data['面积'], data['总价'])
plt.show()
x = data['面积']
y = data['总价']
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.summary()
model.compile(optimizer='adam', loss='mse')
history = model.fit(x, y, epochs=10000)
model.predict(data_test.iloc[:10, -3])
plt.plot(data['面积'], data['总价'], 'bo',
         data['面积'], model.predict(data.iloc[:, -3]), 'ro')
plt.show()
plt.plot(data_test['面积'], data_test['总价'], 'bo',
         data_test['面积'], model.predict(data_test.iloc[:, -3]), 'ro')
plt.show()

#
x2 = data.iloc[:, 2:-1]
y2 = data.iloc[:, -1]

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(30, input_shape=(3,), activation='relu'),
    tf.keras.layers.Dense(30, input_shape=(3,), activation='relu'),
    tf.keras.layers.Dense(30, input_shape=(3,), activation='relu'),
    tf.keras.layers.Dense(30, input_shape=(3,), activation='relu'),
    tf.keras.layers.Dense(1)])
model2.summary()
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse')
history2 = model2.fit(x2, y2, epochs=10000)
model2.predict(data_test.iloc[:10, 2:-1])
plt.plot(data['面积'], data['总价'], 'bo',
         data['面积'], model2.predict(data.iloc[:, 2:-1]), 'ro')
plt.show()
plt.plot(data_test['面积'], data_test['总价'], 'bo',
         data_test['面积'], model2.predict(data_test.iloc[:, 2:-1]), 'ro')
plt.show()
