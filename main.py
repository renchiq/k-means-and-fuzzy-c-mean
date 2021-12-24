import json
import csv
import math
import random

import matplotlib.pyplot as plt

config = {}


def sync_config():
    with open('config.json', 'r') as file:
        global config
        config = json.load(file)

    print('> Config synced.')


def clear_dataset():
    print(' > Started data formatting task.')

    with open(f'data/{config["configuration_variables"]["dataset_file"]}', 'r', encoding='utf-8') as file:
        with open('data/dataset.csv', 'w', newline='', encoding='utf-8', ) as clear_file:
            csv_reader = csv.reader(file)

            fieldnames = ['app_name', 'rating', 'reviews', 'installs']
            csv_writer = csv.DictWriter(clear_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            next(csv_reader, None)

            for row in csv_reader:
                name, rating, reviews, installs = row[0], row[2], row[4], row[5]

                if rating == 'NaN':
                    rating = random.randint(0, 4) + random.randint(0, 9) / 10

                if not reviews == 'Varies with device':
                    if 'M' in reviews:
                        reviews = reviews[:-1]
                    elif 'k' in reviews:
                        # converting to Mil
                        reviews = float(reviews[:-1]) / 1000
                else:
                    continue

                # print(name, installs)
                installs = int(installs.replace('+', '').replace(',', '')) / 1000

                csv_writer.writerow({'app_name': name, 'rating': rating, 'reviews': reviews, 'installs': installs})

    print(' > Ended data formatting task. \033[92mSuccess.\033[0m')


def prepare_data():
    points = list()

    with open(f'data/{config["configuration_variables"]["dataset_file"]}', 'r', encoding='utf-8', ) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)

        for point in csv_reader:
            _, rating, reviews, installs = point
            points.append([float(rating), float(reviews), float(installs)])

    return points


def build_plot(data, _centroids_data=None, filename=None):
    fig = plt.figure(figsize=(12, 12), dpi=260)
    ax = fig.add_subplot(projection='3d')

    print(f' > Started plotting.')

    ax.set_xlabel('installs (mil)')
    ax.set_ylabel('total reviews (mil)')
    ax.set_zlabel('rating')

    if _centroids_data:
        colors = config['configuration_variables']['colors']

        for point in data:
            rating, reviews, installs, affinity = point

            ax.scatter(float(installs), float(reviews), float(rating), color=colors[affinity], marker='o')

        for index, centroid_data in enumerate(_centroids_data):
            rating, reviews, installs = centroid_data
            ax.scatter(float(installs), float(reviews), float(rating), marker='D', color='#000000',
                       edgecolor='#ffffff', alpha=1)
            ax.text(float(installs), float(reviews), float(rating), f'#{index}', size=20, zorder=1, color='k')

    else:
        for point in data:
            rating, reviews, installs = point
            ax.scatter(float(installs), float(reviews), float(rating))

    print(f' > Ended plotting "{filename}". \033[92mSuccess.\033[0m')

    print(f' > Preparing image...')

    if filename:
        plt.savefig(f'results/{filename}.png')

    plt.show()


def build_fuzzy_plot(data, _centroids_data=None, filename=None):
    fig = plt.figure(figsize=(12, 12), dpi=260)
    ax = fig.add_subplot(projection='3d')

    print(f' > Started plotting.')

    ax.set_xlabel('installs (mil)')
    ax.set_ylabel('total reviews (mil)')
    ax.set_zlabel('rating')

    if _centroids_data:
        colors = config['configuration_variables']['colors']

        for point in data:
            rating, reviews, installs, affinity_scores = point
            # print(affinity_scores)

            _alpha = max(affinity_scores) - 0.3
            _color_index = affinity_scores.index(max(affinity_scores))

            ax.scatter(float(installs), float(reviews), float(rating), color=colors[_color_index], marker='o',
                       alpha=_alpha)

        for index, centroid_data in enumerate(_centroids_data):
            rating, reviews, installs = centroid_data
            ax.scatter(float(installs), float(reviews), float(rating), marker='D', color='#000000',
                       edgecolor='#ffffff', alpha=1)
            ax.text(float(installs), float(reviews), float(rating), f'#{index}', size=20, zorder=1, color='k')

    else:
        for point in data:
            rating, reviews, installs = point
            ax.scatter(float(installs), float(reviews), float(rating))

    print(f' > Ended plotting "{filename}". \033[92mSuccess.\033[0m')

    print(f' > Preparing image...')

    if filename:
        plt.savefig(f'results/{filename}.png')

    plt.show()


def set_random_centroids(quantity):
    _centroids_data = list()

    for _ in range(quantity):
        centroid_data = [random.randint(0, 100) / 20, random.randint(10, 90), random.randint(0, 500)]
        _centroids_data.append(centroid_data)

    return _centroids_data


def calculate_distances(_centroids_data, points_data):
    _centroid_results = list()

    for centroid_data in _centroids_data:
        _results = list()

        for point_data in points_data:
            # from centroid to point

            e_distance = (point_data[0] - centroid_data[0]) ** 2
            e_distance += (point_data[1] - centroid_data[1]) ** 2
            e_distance += (point_data[2] - centroid_data[2]) ** 2
            e_distance = round(math.sqrt(e_distance), 2)

            _results.append(e_distance)

        _centroid_results.append(_results)

    return _centroid_results


def calculate_affinity(_centroid_results, points_data):
    points_amount = len(_centroid_results[0])
    centroid_groups_amount = len(_centroid_results)

    for index in range(points_amount):
        heap = list()

        for centroid_group in range(centroid_groups_amount):
            heap.append(_centroid_results[centroid_group][index])

        if len(points_data[index]) == 3:
            points_data[index].append(heap.index(max(heap)))
        else:
            points_data[index][3] = (heap.index(max(heap)))

    return points_data


def calculate_fuzzy_affinity(_centroid_results, points_data):
    points_amount = len(_centroid_results[0])
    centroid_groups_amount = len(_centroid_results)

    for index in range(points_amount):
        heap = list()

        for centroid_group in range(centroid_groups_amount):
            heap.append(_centroid_results[centroid_group][index])

        total_affinity = sum(heap)
        for i in range(len(heap)):
            heap[i] = round(heap[i] / total_affinity, 2)

        if len(points_data[index]) == 3:
            points_data[index].append(heap)
        else:
            points_data[index][3] = heap

    return points_data


def calculate_centroids(points_data):
    result = list()

    for i in range(config['configuration_variables']['centroids']['quantity']):
        result.append({
            'total_points': 0,
            'total_rating': 0,
            'total_reviews': 0,
            'total_installs': 0,
        })

    for point in points_data:
        result[point[3]]['total_rating'] += point[0]
        result[point[3]]['total_reviews'] += point[1]
        result[point[3]]['total_installs'] += point[2]

        result[point[3]]['total_points'] += 1

    new_centroids_data = list()

    for centroid in result:
        total_points = centroid['total_points']

        if total_points != 0:
            new_centroids_data.append([centroid['total_rating'] / total_points,
                                       centroid['total_reviews'] / total_points,
                                       centroid['total_installs'] / total_points])

    # print(new_centroids_data)
    return new_centroids_data


def calculate_fuzzy_centroids(points_data):
    result = list()

    for i in range(config['configuration_variables']['centroids']['quantity']):
        result.append({
            'total_points': 0,
            'total_rating': 0,
            'total_reviews': 0,
            'total_installs': 0,
        })

    for point in points_data:
        affinity_index = point[3].index(max(point[3]))

        result[affinity_index]['total_rating'] += point[0]
        result[affinity_index]['total_reviews'] += point[1]
        result[affinity_index]['total_installs'] += point[2]

        result[affinity_index]['total_points'] += 1

    new_centroids_data = list()

    for index, centroid in enumerate(result):
        total_points = centroid['total_points']

        if total_points != 0:
            new_centroids_data.append([centroid['total_rating'] / total_points,
                                       centroid['total_reviews'] / total_points,
                                       centroid['total_installs'] / total_points])

        print(f'Cluster #{index}: {total_points} total points.')

    return new_centroids_data


def start_hard_clustering():
    print(f'> \033[95mStarting hard clustering task...\033[0m')

    og_data = prepare_data()

    centroids_data = set_random_centroids(config['configuration_variables']['centroids']['quantity'])
    points_distances = list()
    afflicted_data = list()

    for iteration in range(config['configuration_variables']['iterations']):
        points_distances = calculate_distances(centroids_data, og_data)

        afflicted_data = calculate_affinity(points_distances, og_data)

        build_plot(afflicted_data, centroids_data, f'hard-iteration-{iteration}')

        centroids_data = calculate_centroids(afflicted_data)

    print(f'> \033[95mHard clustering ended.\033[0m')


def start_soft_clustering():
    print(f'> \033[95mStarting soft clustering task...\033[0m')

    og_data = prepare_data()

    centroids_data = set_random_centroids(config['configuration_variables']['centroids']['quantity'])
    points_distances = list()
    afflicted_data = list()

    for iteration in range(config['configuration_variables']['iterations']):
        points_distances = calculate_distances(centroids_data, og_data)

        afflicted_data = calculate_fuzzy_affinity(points_distances, og_data)

        build_fuzzy_plot(afflicted_data, centroids_data, f'soft-iteration-{iteration}')

        centroids_data = calculate_fuzzy_centroids(afflicted_data)

    print(f'> \033[95mSoft clustering task ended.\033[0m')


if __name__ == '__main__':
    sync_config()

    # start_hard_clustering()

    start_soft_clustering()
