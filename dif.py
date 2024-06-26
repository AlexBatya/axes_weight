import xml.etree.ElementTree as ET
from scipy.signal import find_peaks
import numpy as np

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.time = []
        self.wes_values = []
    
    def parse_xml(self):
        try:
            # Парсинг XML файла
            tree = ET.parse(self.file_path)
            root = tree.getroot()

            # Получение значений из XML
            self.wes_values = [float(row.attrib['OSWES']) for row in root.findall('.//ROW')]
            self.time = list(range(len(self.wes_values)))  # Временные метки (можно заменить реальными значениями времени)

            # Ограничение до половины данных
            half_index = len(self.wes_values) // 2
            self.wes_values = self.wes_values[:half_index]
            self.time = self.time[:half_index]

            return True
        except ET.ParseError as parse_error:
            print(f'Ошибка при парсинге XML: {parse_error}')
            return False
        except IOError as io_error:
            print(f'Ошибка чтения файла: {io_error}')
            return False
        except Exception as e:
            print(f'Произошла ошибка: {e}')
            return False

    def calculate_derivative(self):
        if not self.time or not self.wes_values:
            return []

        # Вычисление производной
        dt = np.diff(self.time)
        dv = np.diff(self.wes_values)
        derivative = dv / dt
        # Добавление нулевого значения в начало, чтобы размерность совпадала с исходными данными
        derivative = np.insert(derivative, 0, 0)
        return derivative

    @staticmethod
    def moving_average(signal, window_size):
        return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

    @staticmethod
    def count_large_hysteresis(filtered_y_out, threshold=0.1):
        peaks, _ = find_peaks(filtered_y_out, height=threshold)
        hysteresis_count = len(peaks)
        return hysteresis_count, peaks

    @staticmethod
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

    @staticmethod
    def calculate_axle_weights(midpoints):
        axle_weights = [midpoints[0][1]]  # Первый вес оставляем как есть
        for i in range(1, len(midpoints)):
            axle_weight = midpoints[i][1] - midpoints[i - 1][1]
            axle_weights.append(axle_weight)
        return axle_weights

    def process_data(self, threshold=0.5, window_size=11):
        if not self.parse_xml():
            return
        
        derivative = self.calculate_derivative()
        filtered_y_out = self.moving_average(derivative, window_size)
        hysteresis_count, peaks = self.count_large_hysteresis(filtered_y_out, threshold)
        midpoints = self.find_midpoint(peaks, self.wes_values)
        axle_weights = self.calculate_axle_weights(midpoints)
        last_y_value = midpoints[-1][1]
        
        return axle_weights, last_y_value

# Пример использования класса DataProcessor для обработки данных
file_path = './test.xml'
threshold = 80  # Порог для больших положительных гистерезисов
window_size = 11  # Размер окна для скользящего среднего

data_processor = DataProcessor(file_path)
axle_weights, last_y_value = data_processor.process_data(threshold, window_size)

print(data_processor.)

if axle_weights and last_y_value:
    print(f'Веса осей: {axle_weights}')
    print(f'Последнее значение y: {last_y_value}')
