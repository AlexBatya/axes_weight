import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import lsim, find_peaks
import control as ctrl
import numpy as np

def parse_xml(file_path):
    try:
        # Парсинг XML файла
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Получение значений из XML
        wes_values = [float(row.attrib['OSWES']) for row in root.findall('.//ROW')]
        time = list(range(len(wes_values)))  # Временные метки (можно заменить реальными значениями времени)

        # Ограничение до половины данных
        half_index = len(wes_values) // 2
        wes_values = wes_values[:half_index]
        time = time[:half_index]

        return time, wes_values

    except ET.ParseError as parse_error:
        print(f'Ошибка при парсинге XML: {parse_error}')
        return [], []
    except IOError as io_error:
        print(f'Ошибка чтения файла: {io_error}')
        return [], []
    except Exception as e:
        print(f'Произошла ошибка: {e}')
        return [], []

def simulate_system(time, values):
    # Создание дифференцирующего звена (H(s) = s / (s + 1))
    numerator = [1, 0]
    denominator = [1, 5, 1]
    system = ctrl.TransferFunction(numerator, denominator)

    # Печать передаточной функции
    print("Передаточная функция H(s):")
    print(system)

    # Симуляция системы
    t_out, y_out, _ = lsim((numerator, denominator), U=values, T=time)
    
    return t_out, y_out

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def count_large_hysteresis(filtered_y_out, threshold=0.1):
    peaks, _ = find_peaks(filtered_y_out, height=threshold)
    hysteresis_count = len(peaks)
    return hysteresis_count, peaks

def find_midpoint(peaks, wes_values):
    midpoints = []
    for i in range(len(peaks) - 1):
        midpoint_x = (peaks[i] + peaks[i + 1]) // 2
        midpoint_y = wes_values[midpoint_x]
        midpoints.append((midpoint_x, midpoint_y))
    
    # Добавление последней точки
    last_x = len(wes_values) - 1
    last_y = wes_values[last_x]
    midpoints.append((last_x, last_y))
    
    return midpoints

def calculate_axle_weights(midpoints):
    axle_weights = [midpoints[0][1]]  # Первый вес оставляем как есть
    for i in range(1, len(midpoints)):
        axle_weight = midpoints[i][1] - midpoints[i - 1][1]
        axle_weights.append(axle_weight)
    return axle_weights

def plot_results(time, wes_values, t_out_wes, y_out_wes, peaks_wes, threshold):
    # Применение скользящего среднего для сглаживания данных
    window_size = 11
    filtered_y_out_wes = moving_average(y_out_wes, window_size)
    time_filtered_wes = t_out_wes[:len(filtered_y_out_wes)]

    # Построение графиков
    plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2)

    # График входного сигнала WES
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(time, wes_values, linestyle='-', linewidth=1.0)
    ax1.set_title('Входной сигнал WES78')
    ax1.set_xlabel('Индекс')
    ax1.set_ylabel('Значение WES78')
    ax1.grid(True)

    # График выходного сигнала WES после фильтрации
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(time_filtered_wes, filtered_y_out_wes, linestyle='-', linewidth=1.0)
    ax2.plot(time_filtered_wes[peaks_wes], filtered_y_out_wes[peaks_wes], 'rx')  # Отметки пиков
    ax2.set_title(f'Выходной сигнал WES (после скользящего среднего)\n(Порог гистерезиса: {threshold})')
    ax2.set_xlabel('Индекс')
    ax2.set_ylabel('Отклик системы (сглаженный)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print('Графики успешно построены')

def main(file_path, threshold=0.5):
    time, wes_values = parse_xml(file_path)
    if not time or not wes_values:
        return
    
    t_out_wes, y_out_wes = simulate_system(time, wes_values)
    
    # Применение скользящего среднего для сглаживания данных
    window_size = 11
    filtered_y_out_wes = moving_average(y_out_wes, window_size)
    
    # Определение количества больших положительных гистерезисов для WES
    hysteresis_count_wes, peaks_wes = count_large_hysteresis(filtered_y_out_wes, threshold)
    print(f'Количество больших положительных гистерезисов для WES (порог {threshold}): {hysteresis_count_wes}')
    
    # Получение значений x и y для каждого гистерезиса для WES
    midpoints = find_midpoint(peaks_wes, wes_values)
    for x_new, y in midpoints:
        print(f'Между двумя соседними гистерезисами: x_new = {x_new}, y входного сигнала WES: {y}')
    
    # Расчет веса каждой оси
    axle_weights = calculate_axle_weights(midpoints)
    for i, weight in enumerate(axle_weights):
        print(f'Вес оси {i + 1}: {weight}')
    
    # Вывод последнего значения y
    last_y_value = midpoints[-1][1]
    print(f'Последнее значение y: {last_y_value}')
    
    plot_results(time, wes_values, t_out_wes, y_out_wes, peaks_wes, threshold)

# Укажите путь к вашему XML-файлу и порог для больших положительных гистерезисов
file_path = './data/0406-085523.xml'
threshold = 80  # Задайте порог для больших положительных гистерезисов
main(file_path, threshold)

