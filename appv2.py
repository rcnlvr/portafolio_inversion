import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

# Configuración de la página
st.set_page_config(page_title="Analizador de Portafolio", layout="wide", page_icon="📊")
st.sidebar.title("📈 Analizador Cool de Portafolio de Inversión")

# Funciones auxiliares

def calcular_minima_volatilidad_objetivo(returns, target_return=0.10):
    n = returns.shape[1]
    
    # Función objetivo: minimizar la varianza del portafolio
    def portfolio_volatility(weights):
        cov_matrix = returns.cov()
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)

    # Restricciones
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Pesos deben sumar 1
        {'type': 'eq', 'fun': lambda weights: np.dot(weights, returns.mean() * 252) - target_return}  # Rendimiento objetivo anualizado
    ]
    
    # Límites: los pesos deben estar entre 0 y 1
    bounds = tuple((0, 1) for _ in range(n))
    
    # Pesos iniciales iguales
    initial_weights = np.array([1 / n] * n)
    
    # Optimización
    result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x  # Retorna los pesos óptimos


def calcular_riesgo_black_litterman(returns, P, Q, omega, tau=0.05):
    # Cálculo de la matriz de covarianza
    cov_matrix = returns.cov()
    
    # Cálculo de los rendimientos esperados del mercado
    pi = np.dot(cov_matrix, np.mean(returns, axis=0))
    
    # Ajuste de los rendimientos esperados con las opiniones del inversor
    M_inverse = np.linalg.inv(np.dot(tau, cov_matrix))
    omega_inverse = np.linalg.inv(omega)
    adjusted_returns = np.dot(np.linalg.inv(M_inverse + np.dot(P.T, np.dot(omega_inverse, P))), 
                              np.dot(M_inverse, pi) + np.dot(P.T, np.dot(omega_inverse, Q)))
    
    # Cálculo del riesgo ajustado
    adjusted_cov_matrix = cov_matrix + np.dot(np.dot(P.T, omega_inverse), P)
    riesgo = np.sqrt(np.dot(adjusted_returns.T, np.dot(adjusted_cov_matrix, adjusted_returns)))
    
    return riesgo

def calcular_rendimiento_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    return (1 + returns.iloc[-window:]).prod() - 1

def calcular_sesgo(df):
    return df.skew()

def calcular_exceso_curtosis(returns):
    return returns.kurtosis()

def calcular_ultimo_drawdown(series):
    peak = series.expanding(min_periods=1).max()
    drawdown = (series - peak) / peak
    ultimo_drawdown = drawdown.iloc[-1]
    return ultimo_drawdown

def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def calcular_metricas(df):
    returns = df.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def calcular_rendimientos_portafolio(returns, weights):
    return (returns * weights).sum(axis=1)

def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calcular_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else np.nan

def calcular_beta(asset_returns, market_returns):
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance != 0 else np.nan

def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

def calcular_var_cvar_ventana(returns, window):
    if len(returns) < window:
        return np.nan, np.nan
    window_returns = returns.iloc[-window:]
    return calcular_var_cvar(window_returns)

def crear_histograma_distribucion(returns, var_95, cvar_95, title):
    fig = go.Figure()
    counts, bins = np.histogram(returns, bins=50)
    mask_before_var = bins[:-1] <= var_95

    fig.add_trace(go.Bar(
        x=bins[:-1][mask_before_var],
        y=counts[mask_before_var],
        width=np.diff(bins)[mask_before_var],
        name='Retornos < VaR',
        marker_color='rgba(255, 69, 0, 0.8)'  # Rojo intenso
    ))

    fig.add_trace(go.Bar(
        x=bins[:-1][~mask_before_var],
        y=counts[~mask_before_var],
        width=np.diff(bins)[~mask_before_var],
        name='Retornos > VaR',
        marker_color='rgba(50, 205, 50, 0.8)'  # Verde brillante
    ))

    fig.add_trace(go.Scatter(
        x=[var_95, var_95],
        y=[0, max(counts)],
        mode='lines',
        name='VaR 95%',
        line=dict(color='dodgerblue', width=3, dash='dash')  # Azul brillante
    ))

    fig.add_trace(go.Scatter(
        x=[cvar_95, cvar_95],
        y=[0, max(counts)],
        mode='lines',
        name='CVaR 95%',
        line=dict(color='purple', width=3, dash='dot')  # Morado brillante
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='midnightblue')),
        xaxis=dict(title='Retornos', showgrid=True, gridcolor='lightblue', zerolinecolor='blue'),
        yaxis=dict(title='Frecuencia', showgrid=True, gridcolor='lightblue', zerolinecolor='blue'),
        barmode='overlay',
        bargap=0.1,
        plot_bgcolor='rgba(240, 248, 255, 1)',  # Azul claro
        paper_bgcolor='rgba(230, 230, 250, 1)',  # Lavanda
        legend=dict(font=dict(size=12, color='darkblue'))
    )
    return fig


def calcular_minima_varianza(returns):
    n = returns.shape[1]
    
    # Función objetivo: minimizar la varianza
    def portfolio_variance(weights):
        cov_matrix = returns.cov()
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Restricciones: los pesos deben sumar 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Límites: los pesos deben estar entre 0 y 1
    bounds = tuple((0, 1) for _ in range(n))
    
    # Pesos iniciales iguales
    initial_weights = np.array([1 / n] * n)
    
    # Optimización
    result = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x  # Retorna los pesos óptimos

#Portafolio Maximo Sharpe Ratio
def calcular_maximo_sharpe(returns, risk_free_rate=0.02):
    n = returns.shape[1]
    
    # Función objetivo: maximizar el Sharpe Ratio
    def negative_sharpe_ratio(weights):
        portfolio_return = np.dot(weights, returns.mean()) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe_ratio  # Negativo para maximización
    
    # Restricciones: los pesos deben sumar 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Límites: los pesos deben estar entre 0 y 1
    bounds = tuple((0, 1) for _ in range(n))
    
    # Pesos iniciales iguales
    initial_weights = np.array([1 / n] * n)
    
    # Optimización
    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x  #
        


# ETFs permitidos y datos
etfs_permitidos = ["IEI", "EMB", "SPY", "IEMG", "GLD"]
start_date = "2010-01-01"
end_date = "2023-12-31"

simbolos_input = st.sidebar.text_input(
    "🧩 Ingrese los símbolos de los ETFs (IEI, EMB, SPY, IEMG, GLD):", 
    ",".join(etfs_permitidos)
)
pesos_input = st.sidebar.text_input(
    "📊 Ingrese los pesos correspondientes (deben sumar 1):", 
    "0.2,0.2,0.2,0.2,0.2"
)

simbolos = [s.strip() for s in simbolos_input.split(',') if s.strip() in etfs_permitidos]
pesos = [float(w.strip()) for w in pesos_input.split(',')]

# Selección del benchmark
benchmark_options = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI"
}
selected_benchmark = st.sidebar.selectbox("Seleccione el benchmark:", list(benchmark_options.keys()))
benchmark = benchmark_options[selected_benchmark]

if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    st.sidebar.error("El número de símbolos debe coincidir con el número de pesos, y los pesos deben sumar 1.")
else:
    # Obtener datos
    all_symbols = simbolos + [benchmark]
    df_stocks = obtener_datos_acciones(all_symbols, start_date, end_date)
    returns, cumulative_returns, normalized_prices = calcular_metricas(df_stocks)
    
    # Rendimientos del portafolio
    portfolio_returns = calcular_rendimientos_portafolio(returns[simbolos], pesos)
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Crear pestañas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Análisis de Activos Individuales", "Análisis del Portafolio", "Portafolio Mínima Varianza", "Portafolio Max Sharpe Ratio","Portafolio Mínima Vol 10% obj", "Portafolio Black Litterman"])

    etf_summaries = {
        "IEI": {
            "nombre": "iShares 3-7 Year Treasury Bond ETF",
            "exposicion": "Bonos del Tesoro de EE. UU. con vencimientos entre 3 y 7 años",
            "indice": "ICE U.S. Treasury 3-7 Year Bond Index",
            "moneda": "USD",
            "pais": "Estados Unidos",
            "estilo": "Renta fija desarrollada",
            "costos": "0.15%",
        },
        "EMB": {
            "nombre": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
            "exposicion": "Bonos soberanos y cuasi-soberanos de mercados emergentes",
            "indice": "J.P. Morgan EMBI Global Core Index",
            "moneda": "USD",
            "pais": "Diversos mercados emergentes (Brasil, México, Sudáfrica, etc.)",
            "estilo": "Renta fija emergente",
            "costos": "0.39%",
        },
        "SPY": {
            "nombre": "SPDR S&P 500 ETF Trust",
            "exposicion": "500 empresas más grandes de Estados Unidos",
            "indice": "S&P 500 Index",
            "moneda": "USD",
            "pais": "Estados Unidos",
            "estilo": "Renta variable desarrollada",
            "costos": "0.09%",
        },
        "IEMG": {
            "nombre": "iShares Core MSCI Emerging Markets ETF",
            "exposicion": "Empresas de gran y mediana capitalización en mercados emergentes",
            "indice": "MSCI Emerging Markets Investable Market Index",
            "moneda": "USD",
            "pais": "China, India, Brasil, y otros mercados emergentes",
            "estilo": "Renta variable emergente",
            "costos": "0.11%",
        },
        "GLD": {
            "nombre": "SPDR Gold Shares",
            "exposicion": "Precio del oro físico (lingotes almacenados en bóvedas)",
            "indice": "Precio spot del oro",
            "moneda": "USD",
            "pais": "Exposición global",
            "estilo": "Materias primas",
            "costos": "0.40%",
        }
        }

    with tab1:
        
        
        st.header("Análisis de Activos Individuales")
        selected_asset = st.selectbox("Seleccione un ETF para analizar:", simbolos)

        if selected_asset:
            # Resumen del ETF
            st.subheader(f"Resumen del ETF: {selected_asset}")
            summary = etf_summaries[selected_asset]
            st.markdown(f"""
            - **Nombre:** {summary['nombre']}
            - **Exposición:** {summary['exposicion']}
            - **Índice que sigue:** {summary['indice']}
            - **Moneda de denominación:** {summary['moneda']}
            - **País o región principal:** {summary['pais']}
            - **Estilo:** {summary['estilo']}
            - **Costos:** {summary['costos']}
            """)

        # Cálculos métricos
        var_95, cvar_95 = calcular_var_cvar(returns[selected_asset])
        sharpe = calcular_sharpe_ratio(returns[selected_asset])
        sortino = calcular_sortino_ratio(returns[selected_asset])
        sesgo = calcular_sesgo(returns[selected_asset])
        exceso_curtosis = calcular_exceso_curtosis(returns[selected_asset]) 
        ultimo_drawdown = calcular_ultimo_drawdown(cumulative_returns[selected_asset])

        # Mostrar métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Total", f"{cumulative_returns[selected_asset].iloc[-1]:.2%}")
        col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col3.metric("Sortino Ratio", f"{sortino:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("VaR 95%", f"{var_95:.2%}")
        col5.metric("CVaR 95%", f"{cvar_95:.2%}")
        col6.metric("Media Retornos", f"{returns[selected_asset].mean():.2%}")

        col7, col8, col9 = st.columns(3)
        col7.metric("Sesgo de Retornos", f"{sesgo:.3f}")
        col8.metric("Exceso de Curtosis", f"{exceso_curtosis:.3f}")
        col9.metric("Drawdown", f"{ultimo_drawdown:.2%}")

        # Gráficos
        fig_asset = go.Figure()
        fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[selected_asset], name=selected_asset))
        fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[benchmark], name=selected_benchmark))
        fig_asset.update_layout(title=f'Precio Normalizado: {selected_asset} vs {selected_benchmark} (Base 100)', xaxis_title='Fecha', yaxis_title='Precio Normalizado')
        st.plotly_chart(fig_asset, use_container_width=True, key="price_normalized")

        # Beta
        beta_asset = calcular_beta(returns[selected_asset], returns[benchmark])
        st.metric(f"Beta vs {selected_benchmark}", f"{beta_asset:.2f}")

        st.subheader(f"Distribución de Retornos: {selected_asset} vs {selected_benchmark}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma para el activo seleccionado
            var_asset, cvar_asset = calcular_var_cvar(returns[selected_asset])
            fig_hist_asset = crear_histograma_distribucion(
                returns[selected_asset],
                var_asset,
                cvar_asset,
                f'Distribución de Retornos - {selected_asset}'
            )
            st.plotly_chart(fig_hist_asset, use_container_width=True, key="hist_asset")
            
        with col2:
            # Histograma para el benchmark
            var_bench, cvar_bench = calcular_var_cvar(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                cvar_bench,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_1")
    
    with tab2:
        st.header("Análisis del Portafolio")
        
        # Calcular VaR y CVaR para el portafolio
        portfolio_var_95, portfolio_cvar_95 = calcular_var_cvar(portfolio_returns)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Total del Portafolio", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
        col2.metric("Sharpe Ratio del Portafolio", f"{calcular_sharpe_ratio(portfolio_returns):.2f}")
        col3.metric("Sortino Ratio del Portafolio", f"{calcular_sortino_ratio(portfolio_returns):.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("VaR 95% del Portafolio", f"{portfolio_var_95:.2%}")
        col5.metric("CVaR 95% del Portafolio", f"{portfolio_cvar_95:.2%}")
        col6.metric("Media retornos", f"{portfolio_returns.mean():.2f}")
        
        col7, col8, col9 = st.columns(3)
        col7.metric("Sesgo del Portafolio", f"{calcular_sesgo(portfolio_returns):.2f}")
        col8.metric("Exceso de Curtosis Portafolio", f"{calcular_exceso_curtosis(portfolio_returns):.2f}")
        col9.metric("Drawdown", f"{calcular_ultimo_drawdown(portfolio_returns):.2%}")

        # Gráfico de rendimientos acumulados del portafolio vs benchmark
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns, name='Portafolio'))
        fig_cumulative.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[benchmark], name=selected_benchmark))
        fig_cumulative.update_layout(title=f'Rendimientos Acumulados: Portafolio vs {selected_benchmark}', xaxis_title='Fecha', yaxis_title='Rendimiento Acumulado')
        st.plotly_chart(fig_cumulative, use_container_width=True, key="cumulative_returns")


        # Beta del portafolio vs benchmark
        beta_portfolio = calcular_beta(portfolio_returns, returns[benchmark])
        st.metric(f"Beta del Portafolio vs {selected_benchmark}", f"{beta_portfolio:.2f}")
        st.subheader("Distribución de Retornos del Portafolio vs Benchmark")
        
        col1, col2 = st.columns(2)
            
        with col1:
            # Histograma para el portafolio
            var_port, cvar_port = calcular_var_cvar(portfolio_returns)
            fig_hist_port = crear_histograma_distribucion(
                portfolio_returns,
                var_port,
                cvar_port,
                'Distribución de Retornos - Portafolio'
            )
            st.plotly_chart(fig_hist_port, use_container_width=True, key="hist_port")
            
        with col2:
            # Histograma para el benchmark
            var_bench, cvar_bench = calcular_var_cvar(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                cvar_bench,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_2")

        # Rendimientos y métricas de riesgo en diferentes ventanas de tiempo
        st.subheader("Rendimientos y Métricas de Riesgo en Diferentes Ventanas de Tiempo")
        ventanas = [1, 7, 30, 90, 180, 252]
        
        # Crear DataFrames separados para cada métrica
        rendimientos_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        var_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        cvar_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        
        for ventana in ventanas:
            # Rendimientos
            rendimientos_ventanas[f'{ventana}d'] = pd.Series({
                'Portafolio': calcular_rendimiento_ventana(portfolio_returns, ventana),
                **{symbol: calcular_rendimiento_ventana(returns[symbol], ventana) for symbol in simbolos},
                selected_benchmark: calcular_rendimiento_ventana(returns[benchmark], ventana)
            })
            
            # VaR y CVaR
            var_temp = {}
            cvar_temp = {}
            
            # Para el portafolio
            port_var, port_cvar = calcular_var_cvar_ventana(portfolio_returns, ventana)
            var_temp['Portafolio'] = port_var
            cvar_temp['Portafolio'] = port_cvar
            
            # Para cada símbolo
            for symbol in simbolos:
                var, cvar = calcular_var_cvar_ventana(returns[symbol], ventana)
                var_temp[symbol] = var
                cvar_temp[symbol] = cvar
            
            # Para el benchmark
            bench_var, bench_cvar = calcular_var_cvar_ventana(returns[benchmark], ventana)
            var_temp[selected_benchmark] = bench_var
            cvar_temp[selected_benchmark] = bench_cvar
            
            var_ventanas[f'{ventana}d'] = pd.Series(var_temp)
            cvar_ventanas[f'{ventana}d'] = pd.Series(cvar_temp)
        
        # Mostrar las tablas
        st.subheader("Rendimientos por Ventana")
        st.dataframe(rendimientos_ventanas.style.format("{:.2%}"))
        
        st.subheader("VaR 95% por Ventana")
        st.dataframe(var_ventanas.style.format("{:.2%}"))
        
        st.subheader("CVaR 95% por Ventana")
        st.dataframe(cvar_ventanas.style.format("{:.2%}"))

        # Gráfico de comparación de rendimientos
        fig_comparison = go.Figure()
        for index, row in rendimientos_ventanas.iterrows():
            fig_comparison.add_trace(go.Bar(x=ventanas, y=row, name=index))
        fig_comparison.update_layout(title='Comparación de Rendimientos', xaxis_title='Días', yaxis_title='Rendimiento', barmode='group')
        # Gráfico de comparación de rendimientos
        st.plotly_chart(fig_comparison, use_container_width=True, key="returns_comparison")



with tab3:
    st.header("Análisis del Portafolio de Mínima Varianza")
    
    # Calcular los pesos óptimos
    min_var_weights = calcular_minima_varianza(returns[simbolos])
    
    # Calcular métricas del portafolio de mínima varianza
    min_var_returns = calcular_rendimientos_portafolio(returns[simbolos], min_var_weights)
    min_var_cumulative = (1 + min_var_returns).cumprod() - 1
    min_var_risk = np.sqrt(252) * min_var_returns.std()
    min_var_mean_return = min_var_returns.mean() * 252  # Anualizado
    
    st.subheader("Pesos del Portafolio de Mínima Varianza")
    weights_df = pd.DataFrame({
        "ETF": simbolos,
        "Peso Óptimo": min_var_weights
    })
    st.dataframe(weights_df.style.format({"Peso Óptimo": "{:.2%}"}))
    
    # Mostrar métricas clave
    col1, col2, col3 = st.columns(3)
    col1.metric("Riesgo (Desviación Estándar Anualizada)", f"{min_var_risk:.2%}")
    col2.metric("Rendimiento Esperado Anualizado", f"{min_var_mean_return:.2%}")    
    
    # Comparar rendimientos acumulados
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=min_var_cumulative.index, 
        y=min_var_cumulative, 
        name="Portafolio de Mínima Varianza",
        line=dict(color='royalblue')
    ))
    fig_cumulative.add_trace(go.Scatter(
        x=portfolio_cumulative_returns.index, 
        y=portfolio_cumulative_returns, 
        name="Portafolio Actual",
        line=dict(color='orange', dash='dot')
    ))
    fig_cumulative.add_trace(go.Scatter(
        x=cumulative_returns.index, 
        y=cumulative_returns[benchmark], 
        name=f"Benchmark: {selected_benchmark}",
        line=dict(color='green', dash='dash')
    ))
    fig_cumulative.update_layout(
        title="Comparación de Rendimientos Acumulados",
        xaxis_title="Fecha",
        yaxis_title="Rendimientos Acumulados",
        plot_bgcolor='rgba(240,240,240,1)'
    )
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Distribución de rendimientos del portafolio de mínima varianza
    var_95, cvar_95 = calcular_var_cvar(min_var_returns)
    fig_dist = crear_histograma_distribucion(
        min_var_returns,
        var_95,
        cvar_95,
        title="Distribución de Retornos del Portafolio de Mínima Varianza"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab4:
    st.header("Análisis del Portafolio de Máximo Sharpe Ratio")
    
    # Calcular los pesos óptimos
    max_sharpe_weights = calcular_maximo_sharpe(returns[simbolos])
    
    # Calcular métricas del portafolio de máximo Sharpe Ratio
    max_sharpe_returns = calcular_rendimientos_portafolio(returns[simbolos], max_sharpe_weights)
    max_sharpe_cumulative = (1 + max_sharpe_returns).cumprod() - 1
    max_sharpe_risk = np.sqrt(252) * max_sharpe_returns.std()
    max_sharpe_mean_return = max_sharpe_returns.mean() * 252  # Anualizado
    risk_free_rate = 0.02
    max_sharpe_ratio = (max_sharpe_mean_return - risk_free_rate) / max_sharpe_risk
    
    st.subheader("Pesos del Portafolio de Máximo Sharpe Ratio")
    weights_df = pd.DataFrame({
        "ETF": simbolos,
        "Peso Óptimo": max_sharpe_weights
    })
    st.dataframe(weights_df.style.format({"Peso Óptimo": "{:.2%}"}))
    
    # Mostrar métricas clave
    col1, col2, col3 = st.columns(3)
    col1.metric("Riesgo (Desviación Estándar Anualizada)", f"{max_sharpe_risk:.2%}")
    col2.metric("Rendimiento Esperado Anualizado", f"{max_sharpe_mean_return:.2%}")
    col3.metric("Sharpe Ratio", f"{max_sharpe_ratio:.2f}")
    
    # Comparar rendimientos acumulados
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=max_sharpe_cumulative.index, 
        y=max_sharpe_cumulative, 
        name="Portafolio de Máximo Sharpe Ratio",
        line=dict(color='gold')
    ))
    fig_cumulative.add_trace(go.Scatter(
        x=portfolio_cumulative_returns.index, 
        y=portfolio_cumulative_returns, 
        name="Portafolio Actual",
        line=dict(color='orange', dash='dot')
    ))
    fig_cumulative.add_trace(go.Scatter(
        x=cumulative_returns.index, 
        y=cumulative_returns[benchmark], 
        name=f"Benchmark: {selected_benchmark}",
        line=dict(color='green', dash='dash')
    ))
    fig_cumulative.update_layout(
        title="Comparación de Rendimientos Acumulados",
        xaxis_title="Fecha",
        yaxis_title="Rendimientos Acumulados",
        plot_bgcolor='rgba(240,240,240,1)'
    )
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Distribución de rendimientos del portafolio de máximo Sharpe Ratio
    var_95, cvar_95 = calcular_var_cvar(max_sharpe_returns)
    fig_dist = crear_histograma_distribucion(
        max_sharpe_returns,
        var_95,
        cvar_95,
        title="Distribución de Retornos del Portafolio de Máximo Sharpe Ratio"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab5:
    st.header("Portafolio de Mínima Volatilidad con Objetivo de Rendimiento (MXN)")

    # Convertir los rendimientos a pesos mexicanos suponiendo un tipo de cambio simulado
    tipo_cambio_usd_mxn = 17.0  # Puedes actualizar el tipo de cambio según sea necesario
    returns_mxn = returns[simbolos] * tipo_cambio_usd_mxn
    
    # Calcular los pesos óptimos para el portafolio de mínima volatilidad con un rendimiento objetivo del 10%
    min_vol_weights = calcular_minima_volatilidad_objetivo(returns_mxn)

    # Calcular métricas del portafolio de mínima volatilidad con rendimiento objetivo
    min_vol_returns = calcular_rendimientos_portafolio(returns_mxn, min_vol_weights)
    min_vol_cumulative = (1 + min_vol_returns).cumprod() - 1
    min_vol_risk = np.sqrt(252) * min_vol_returns.std()
    min_vol_mean_return = min_vol_returns.mean() * 252  # Anualizado

    st.subheader("Pesos del Portafolio de Mínima Volatilidad con Objetivo de Rendimiento")
    weights_df = pd.DataFrame({
        "ETF": simbolos,
        "Peso Óptimo": min_vol_weights
    })
    st.dataframe(weights_df.style.format({"Peso Óptimo": "{:.2%}"}))

    # Mostrar métricas clave
    col1, col2 = st.columns(2)
    col1.metric("Riesgo (Desviación Estándar Anualizada)", f"{min_vol_risk:.2%}")
    col2.metric("Rendimiento Esperado Anualizado", f"{min_vol_mean_return:.2%}")

    # Comparar rendimientos acumulados
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=min_vol_cumulative.index, 
        y=min_vol_cumulative, 
        name="Portafolio de Mínima Volatilidad con Objetivo",
        line=dict(color='blue')
    ))
    fig_cumulative.add_trace(go.Scatter(
        x=portfolio_cumulative_returns.index, 
        y=portfolio_cumulative_returns, 
        name="Portafolio Actual",
        line=dict(color='orange', dash='dot')
    ))
    fig_cumulative.add_trace(go.Scatter(
        x=cumulative_returns.index, 
        y=cumulative_returns[benchmark], 
        name=f"Benchmark: {selected_benchmark}",
        line=dict(color='green', dash='dash')
    ))
    fig_cumulative.update_layout(
        title="Comparación de Rendimientos Acumulados",
        xaxis_title="Fecha",
        yaxis_title="Rendimientos Acumulados",
        plot_bgcolor='rgba(240,240,240,1)'
    )
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
   
    
    

with tab6:
    st.title('Optimización con el Modelo de Black-Litterman')
    returns = pd.DataFrame(cumulative_returns[simbolos])
    P = np.array([[1, -1, 0], [0, 1, -1]])
    Q = np.array([0.01, 0.02])
    omega = np.diag([0.0001, 0.0001])
    
    riesgo = calcular_riesgo_black_litterman(returns, P, Q, omega)
    st.write(f'El riesgo ajustado calculado con el modelo Black-Litterman es: {riesgo}')
