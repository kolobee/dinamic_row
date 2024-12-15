import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit

# Функции для обработки данных
def moving_average(data, L):
    smoothed = data.copy()
    for i in range(1, len(data) - 1):
        if i < L:
            smoothed[i] = np.mean(data[:i + L])
        elif i > len(data) - L:
            smoothed[i] = np.mean(data[i - L:])
        else:
            smoothed[i] = np.mean(data[i - L:i + L])
    return smoothed

def exponential_smoothing(data, alpha):
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

def poly2(t, a0, a1, a2):
    return a0 + a1 * t + a2 * (t ** 2)

def fourier_series(t, a0, a1, b1, a2, b2):
    return a0 + a1 * np.cos(2 * np.pi * t / len(t)) + b1 * np.sin(2 * np.pi * t / len(t)) + a2 * np.cos(
        4 * np.pi * t / len(t)) + b2 * np.sin(4 * np.pi * t / len(t))


def calculate_seasonal_component(y_values, t_values, trend_values, max_harmonics=5):

    # Шаг 1: Исключаем тренд
    detrended_values = y_values - trend_values

    # Шаг 2: Период и частоты
    n = len(t_values)
    T = t_values[-1] - t_values[0]
    omega = 2 * np.pi / T

    # Шаг 3: Вычисление коэффициентов Фурье
    a0 = np.mean(detrended_values)
    harmonics = []

    for k in range(1, max_harmonics + 1):
        ak = (2 / n) * np.sum(detrended_values * np.cos(omega * k * t_values))
        bk = (2 / n) * np.sum(detrended_values * np.sin(omega * k * t_values))
        harmonics.append((ak, bk, k))

    # Шаг 4: Формирование сезонной компоненты
    seasonal_component = np.full_like(t_values, a0, dtype=np.float64)
    for ak, bk, k in harmonics:
        seasonal_component += ak * np.cos(omega * k * t_values) + bk * np.sin(omega * k * t_values)

    return seasonal_component


# Добавляем функцию для проверки случайности на основе критерия пиков
def check_randomness(E_t):
    n = len(E_t)
    # Определение пиков
    P = 0
    for i in range(1, n - 1):
        if (E_t[i - 1] < E_t[i] > E_t[i + 1]) or (E_t[i - 1] > E_t[i] < E_t[i + 1]):
            P += 1

    # Математическое ожидание числа пиков
    p = (2 / 3) * (n - 2)
    # Дисперсия числа пиков
    sp2 = (16 * n - 29) / 90
    # Проверка условия независимости
    if P > (p - 1.96 * np.sqrt(sp2)):
        result = "Ряд остатков является случайным"
    else:
        result = "Ряд остатков не является случайным"
    return P, p, sp2, result

def full_dynamic_model(t, popt, popt_fourier):
    trend = poly2(t, *popt)  # Использование оптимальных параметров для полинома второго порядка
    seasonal = fourier_series(t, *popt_fourier)  # Сезонная компонента
    return trend + seasonal

def load_data():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
        # Разбор файла с учетом двоеточий
        data = {int(line.split(":")[0].strip()): float(line.split(":")[1].strip()) for line in lines}
        process_data(data)
    except Exception as e:
        messagebox.showerror("Ошибка загрузки данных", f"Не удалось загрузить данные: {e}")

def process_data(data):
    ti, profit = zip(*data.items())
    df = pd.DataFrame({"ti": ti, "profit": profit})

    # Параметры модели
    L = 2
    alpha = 2 / (len(data) + 1)

    # Модели
    moving_avg = moving_average(df["profit"].values, L)
    exp_smooth = exponential_smoothing(df["profit"].values, alpha)
    popt, _ = curve_fit(poly2, df["ti"].values, df["profit"].values)
    poly_trend = poly2(df["ti"].values, *popt)
    popt_fourier, _ = curve_fit(fourier_series, df["ti"], df["profit"] - poly_trend)
    seasonal_component = fourier_series(df["ti"], *popt_fourier)

    # Вычисление сезонной компоненты с использованием гармонического анализа
    seasonal_component_harmonic = calculate_seasonal_component(
        y_values=df["profit"].values,
        t_values=df["ti"].values,
        trend_values=poly_trend,  # Полиномиальный тренд как трендовая компонента
        max_harmonics=5  # Максимум 5 гармоник
    )

    # Случайная составляющая
    random_component = df["profit"] - (poly_trend + seasonal_component)

    # Проверка случайности
    P, p, sp2, randomness_result = check_randomness(random_component)

    # Полная модель
    full_model = poly_trend + seasonal_component

    # Прогнозы на t = 64, 66, 68, 70
    forecast_times = np.array([64, 66, 68, 70])
    forecast_values = full_dynamic_model(forecast_times, popt, popt_fourier)

    # Таблицы
    table = pd.DataFrame({
        "ti": df["ti"],
        "Исзод. данные": df["profit"],
        "Скольз. ср. (L=2)": moving_avg,
        "Эксп. сгл-е (alpha=0.3)": exp_smooth,
        "Полином. тренд (2 пор-ка)": poly_trend,
        "Сезон. комп-та (Фурье)": seasonal_component,
        "Случ. комп-та": random_component,
        "Полная модель (тренд + сезон)": full_model
    })

    forecast_table = pd.DataFrame({
        "Forecast Time (ti)": forecast_times,
        "Прогноз": forecast_values
    })

    # R^2 для моделей
    R_moving_avg = R_squared(df["profit"].values, moving_avg)
    R_exp_smooth = R_squared(df["profit"].values, exp_smooth)
    R_poly_trend = R_squared(df["profit"].values, poly_trend)
    R_full_model = R_squared(df["profit"], full_model)

    # Вывод данных на вкладки
    add_tab_with_text("Таблица данных", table.to_string(index=False))
    add_tab_with_text("Прогнозы", forecast_table.to_string(index=False))
    add_tab_with_text("R^2 моделей", f"""
Скользящее среднее: R^2 = {R_moving_avg}
Экспоненциальное сглаживание: R^2 = {R_exp_smooth}
Полиномиальный тренд: R^2 = {R_poly_trend}
Точность модели: R^2 = {R_full_model}
    """)
    add_tab_with_text("Случайность ряда", f"""
Общее количество пиков: {P}
Математическое ожидание числа пиков: {p}
Дисперсия числа пиков: {sp2}
Результат проверки случайности: {randomness_result}
    """)

    # Построение графиков
    plot_graphs(df, moving_avg, exp_smooth, poly_trend, full_model, forecast_times, forecast_values, seasonal_component, seasonal_component_harmonic)

def R_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def add_tab_with_text(title, text):
    tab = ttk.Frame(tab_control)
    tab_control.add(tab, text=title)
    text_widget = tk.Text(tab, wrap=tk.WORD, height=10)
    text_widget.insert(tk.END, text)
    text_widget.config(state=tk.DISABLED)  # Запрет редактирования
    text_widget.pack(expand=1, fill="both")

def plot_graphs(df, moving_avg, exp_smooth, poly_trend, full_model, forecast_times, forecast_values, seasonal_component, seasonal_component_harmonic):
    global toggle_original_data
    toggle_original_data = True

    R_moving_avg = R_squared(df["profit"].values, moving_avg)
    R_exp_smooth = R_squared(df["profit"].values, exp_smooth)
    R_poly_trend = R_squared(df["profit"].values, poly_trend)
    R_poly_trend = R_squared(df["profit"].values, poly_trend)

    print("\nСравнение моделей по квадратичному критерию R^2:")
    print(f"Скользящее среднее: R^2 = {R_moving_avg}")
    print(f"Экспоненциальное сглаживание: R^2 = {R_exp_smooth}")
    print(f"Полиномиальный тренд: R^2 = {R_poly_trend}")

    def toggle_data_visibility():
        global toggle_original_data
        toggle_original_data = not toggle_original_data

        # Найти линию, соответствующую оригинальным данным
        for line in ax4.get_lines():
            if line.get_label() == "Исходные данные":
                line.set_visible(toggle_original_data)

        # Перерисовать график
        canvas4.draw()

    def toggle_original_data_moving_avg():
        global toggle_original_data
        toggle_original_data = not toggle_original_data

        # Найти линию, соответствующую оригинальным данным
        for line in ax2.get_lines():
            if line.get_label() == "Исходные данные":
                line.set_visible(toggle_original_data)

        # Перерисовать график
        canvas2.draw()

    def toggle_original_data_exp_smooth():
        global toggle_original_data
        toggle_original_data = not toggle_original_data

        # Найти линию, соответствующую оригинальным данным
        for line in ax3.get_lines():
            if line.get_label() == "Исходные данные":
                line.set_visible(toggle_original_data)

        # Перерисовать график
        canvas3.draw()

    def toggle_original_data_poly_trend():
        global toggle_original_data
        toggle_original_data = not toggle_original_data

        # Найти линию, соответствующую оригинальным данным
        for line in ax_poly.get_lines():
            if line.get_label() == "Исходные данные":
                line.set_visible(toggle_original_data)

        # Перерисовать график
        canvas_poly.draw()


    def toggle_original_data_seasonal():
        global toggle_original_data
        toggle_original_data = not toggle_original_data

        # Найти линию, соответствующую оригинальным данным
        for line in ax_seasonal.get_lines():
            if line.get_label() == "Исходные данные":
                line.set_visible(toggle_original_data)

        # Перерисовать график
        canvas_seasonal.draw()

    def create_interactive_canvas(fig, ax, x, y):
        """Создаёт интерактивный график с отображением данных."""
        canvas = FigureCanvasTkAgg(fig, master=tab_control)
        tooltip = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                              textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                              arrowprops=dict(arrowstyle="->"))
        tooltip.set_visible(False)

        def show_data_on_hover(event):
            for line in ax.get_lines():
                if line.contains(event)[0]:
                    x_data, y_data = line.get_data()
                    idx = (np.abs(x_data - event.xdata)).argmin()
                    tooltip.set_text(f"x: {x_data[idx]:.0f}, y: {y_data[idx]:.2f}")
                    tooltip.xy = (x_data[idx], y_data[idx])
                    tooltip.set_visible(True)
                    canvas.draw()
                    return
            tooltip.set_visible(False)
            canvas.draw()

        canvas.mpl_connect("motion_notify_event", show_data_on_hover)
        return canvas

    # Создание вкладок
    tab_control = ttk.Notebook(app)
    tab1 = ttk.Frame(tab_control)
    tab2 = ttk.Frame(tab_control)
    tab3 = ttk.Frame(tab_control)
    # Вкладка для полиномиального тренда
    tab_poly_trend = ttk.Frame(tab_control)
    tab_seasonal = ttk.Frame(tab_control)
    tab_total = ttk.Frame(tab_control)
    tab4 = ttk.Frame(tab_control)
    tab_control.add(tab1, text="Исходные данные")
    tab_control.add(tab2, text="Скользящее среднее")
    tab_control.add(tab3, text="Экспоненциальное сглаживание")
    tab_control.add(tab_poly_trend, text="Полиномиальный тренд")
    tab_control.add(tab_total, text="Общий график")
    tab_control.add(tab_seasonal, text="Сезонная компонента")
    tab_control.add(tab4, text="Полная модель")
    tab_control.pack(expand=1, fill="both")


    # График 1: Оригинальные данные
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["ti"], df["profit"], label="Исходные данные", color="blue", marker="o")
    ax1.set_title("Исходные данные")
    ax1.set_xlabel("Время (ti)")
    ax1.set_ylabel("Значение")
    ax1.legend()
    ax1.grid()
    ax1.set_xticks(np.arange(min(df["ti"]), max(df["ti"]) + 1, 2))
    ax1.set_yticks(np.linspace(min(df["profit"]), max(df["profit"]), 10))
    canvas1 = create_interactive_canvas(fig1, ax1, df["ti"], df["profit"])
    canvas1.get_tk_widget().pack(in_=tab1)

    # График 2: Скользящее среднее
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df["ti"], moving_avg, label="Скользящее среднее", color="green", linestyle="--")
    ax2.plot(df["ti"], df["profit"], label="Исходные данные", color="blue", marker="o")
    ax2.set_title("Скользящее среднее")
    ax2.set_xlabel("Время (ti)")
    ax2.set_ylabel("Значение")
    ax2.legend()
    ax2.grid()
    ax2.set_xticks(np.arange(min(df["ti"]), max(df["ti"]) + 1, 2))
    ax2.set_yticks(np.linspace(min(moving_avg), max(moving_avg), 10))
    canvas2 = create_interactive_canvas(fig2, ax2, df["ti"], moving_avg)
    canvas2.get_tk_widget().pack(in_=tab2)

    # График 3: Экспоненциальное сглаживание
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(df["ti"], df["profit"], label="Исходные данные", color="blue", marker="o")
    ax3.plot(df["ti"], exp_smooth, label="Экспоненциальное сглаживание", color="red", linestyle="--")
    ax3.set_title("Экспоненциальное сглаживание")
    ax3.set_xlabel("Время (ti)")
    ax3.set_ylabel("Значение")
    ax3.legend()
    ax3.grid()
    ax3.set_xticks(np.arange(min(df["ti"]), max(df["ti"]) + 1, 2))
    ax3.set_yticks(np.linspace(min(exp_smooth), max(exp_smooth), 10))
    canvas3 = create_interactive_canvas(fig3, ax3, df["ti"], exp_smooth)
    canvas3.get_tk_widget().pack(in_=tab3)

    # Построение графика полиномиального тренда
    fig_poly, ax_poly = plt.subplots(figsize=(10, 4))
    ax_poly.plot(df["ti"], df["profit"], label="Исходные данные", color="blue", marker="o")
    ax_poly.plot(df["ti"], poly_trend, label="Полиномиальный тренд второго порядка", color="green", linestyle="--")
    ax_poly.set_title("Полиномиальный тренд второго порядка")
    ax_poly.set_xlabel("Время (ti)")
    ax_poly.set_ylabel("Значение")
    ax_poly.legend()
    ax_poly.grid()
    ax_poly.set_xticks(np.arange(min(df["ti"]), max(df["ti"]) + 1, 2))
    ax_poly.set_yticks(np.linspace(min(poly_trend), max(poly_trend), 10))
    canvas_poly = create_interactive_canvas(fig_poly, ax_poly, df["ti"], df["profit"])
    canvas_poly.get_tk_widget().pack(in_=tab_poly_trend)

    #сезонная компонента
    fig_seasonal, ax_seasonal = plt.subplots(figsize=(10, 5))
    ax_seasonal.plot(df["ti"], seasonal_component, label="Сезонная компонента", color="orange", linestyle="-")
    ax_seasonal.plot(df["ti"], seasonal_component_harmonic, label="Сезонная компонента (гармонический анализ)",
                              color="purple", linestyle="-")
    ax_seasonal.set_title("Сезонная компонента (Фурье)")
    ax_seasonal.set_xlabel("Время (ti)")
    ax_seasonal.set_ylabel("Значение")
    ax_seasonal.legend()
    ax_seasonal.grid()
    ax_seasonal.set_xticks(np.arange(min(df["ti"]), max(df["ti"]) + 1, 2))
    ax_seasonal.set_yticks(np.linspace(min(seasonal_component), max(seasonal_component), 10))
    canvas_seasonal = create_interactive_canvas(fig_seasonal, ax_seasonal, df["ti"], df["profit"])
    canvas_seasonal.get_tk_widget().pack(in_=tab_seasonal)

    # fig_seasonal_harmonic, ax_seasonal_harmonic = plt.subplots(figsize=(10, 5))
    # ax_seasonal_harmonic.plot(df["ti"], seasonal_component_harmonic, label="Сезонная компонента (гармонический анализ)",
    #                           color="purple", linestyle="-")
    # ax_seasonal_harmonic.set_title("Сезонная компонента (гармонический анализ)")
    # ax_seasonal_harmonic.set_xlabel("Время (ti)")
    # ax_seasonal_harmonic.set_ylabel("Значение")
    # ax_seasonal_harmonic.legend()
    # ax_seasonal_harmonic.grid()
    # canvas_seasonal_harmonic = create_interactive_canvas(fig_seasonal_harmonic, ax_seasonal_harmonic, df["ti"],
    #                                                      seasonal_component_harmonic)
    # canvas_seasonal_harmonic.get_tk_widget().pack(in_=tab_seasonal)

    # общий График:
    fig_total, ax_total = plt.subplots(figsize=(10, 4))
    ax_total.plot(df["ti"], df["profit"], label="Исходные данные", color="blue", marker="o")
    ax_total.plot(df["ti"], moving_avg, label="Скользящее среднее", color="green", linestyle="--")
    ax_total.plot(df["ti"], exp_smooth, label="Экспоненциальное сглаживание", color="red", linestyle="--")
    ax_total.plot(df["ti"], poly_trend, label="Полиномиальный тренд второго порядка", color="green", linestyle="--")
    ax_total.set_title("Общий график")
    ax_total.set_xlabel("Время (ti)")
    ax_total.set_ylabel("Значение")
    ax_total.legend()
    ax_total.grid()
    ax_total.set_xticks(np.arange(min(df["ti"]), max(df["ti"]) + 1, 2))
    ax_total.set_yticks(np.linspace(min(df["profit"]), max(df["profit"]), 10))
    canvas_total = create_interactive_canvas(fig_total, ax_total, df["ti"], df["profit"])
    canvas_total.get_tk_widget().pack(in_=tab_total)


    # График 4: Полная модель
    fig4, ax4 = plt.subplots(figsize=(10, 4))

    # Оригинальные данные
    ax4.plot(df["ti"], df["profit"], label="Исходные данные", color="blue", marker="o")
    # Полная модель
    ax4.plot(df["ti"], full_model, label="Полная модель", color="purple", linestyle="-")
    # Случайная составляющая
    random_component = df["profit"] - full_model
    ax4.plot(df["ti"], random_component, label="Случайная компонента", color="red", linestyle="--")
    # Прогноз
    ax4.plot(forecast_times, forecast_values, label="Прогноз", color="orange", marker="x", linestyle="--")

    # Оформление графика
    ax4.set_title("Полная модель и случайная составляющая")
    ax4.set_xlabel("Время (ti)")
    ax4.set_ylabel("Значение")
    ax4.legend()
    ax4.grid()
    ax4.set_xticks(np.arange(min(df["ti"]), max(df["ti"]) + 1, 2))
    ax4.set_yticks(np.linspace(min(full_model), max(full_model), 10))
    canvas4 = create_interactive_canvas(fig4, ax4, df["ti"], df["profit"])
    canvas4.get_tk_widget().pack(in_=tab4)


    # Кнопка для переключения видимости данных
    toggle_button = tk.Button(tab4, text="Показать/Скрыть оригинальные данные", command=toggle_data_visibility)
    toggle_button.pack()
    toggle_button_moving_avg = tk.Button(tab2, text="Показать/Скрыть оригинальные данные", command=toggle_original_data_moving_avg)
    toggle_button_moving_avg.pack()
    toggle_button_exp_smooth = tk.Button(tab3, text="Показать/Скрыть оригинальные данные", command=toggle_original_data_exp_smooth)
    toggle_button_exp_smooth.pack()
    toggle_button_poly_trend = tk.Button(tab_poly_trend, text="Показать/Скрыть оригинальные данные", command=toggle_original_data_poly_trend)
    toggle_button_poly_trend.pack()
    toggle_button_seasonal = tk.Button(tab_seasonal, text="Показать/Скрыть оригинальные данные",
                                       command=toggle_original_data_seasonal)
    toggle_button_seasonal.pack()

    pass

# Создание главного окна приложения
app = tk.Tk()
app.title("Анализ данных")
app.geometry("1280x720")

# Главное окно с вкладками
tab_control = ttk.Notebook(app)
tab_control.pack(expand=1, fill="both")

# Верхний текст и кнопка
label = tk.Label(app, text="Анализ динамических данных", font=("Arial", 16))
label.pack(pady=1)

# Кнопка загрузки данных
button_load = tk.Button(app, text="Загрузить данные", command=load_data, font=("Arial", 14))
button_load.pack(pady=1)

app.protocol("WM_DELETE_WINDOW", app.quit)  # Закрытие программы при нажатии на крестик

app.mainloop()
