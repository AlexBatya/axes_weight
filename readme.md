# DataProcessor

DataProcessor - это программное обеспечение, служащее математической моделью для определения количества осей автомобиля и расчета веса, приходящегося на каждую ось по ступенчатому графику. Программа написана на языке Python и включает парсинг XML, вычисление производной, скользящее среднее и анализ гистерезиса для определения весов осей.

## Основные функции

- **Парсинг XML:** Извлечение данных из XML-файла, содержащего информацию о весах.
- **Вычисление производной:** Расчет производной весовых данных по времени.
- **Скользящее среднее:** Применение фильтра скользящего среднего для сглаживания данных.
- **Анализ гистерезиса:** Определение пиков в данных и расчет количества гистерезисов.
- **Вычисление весов осей:** Определение веса, приходящегося на каждую ось, на основе средних точек между пиками.

## Использование

### Зависимости

Для запуска программы необходимо установить следующие библиотеки:

- xml.etree.ElementTree
- scipy
- numpy

Вы можете установить необходимые зависимости с помощью pip:

```sh
pip install numpy scipy
