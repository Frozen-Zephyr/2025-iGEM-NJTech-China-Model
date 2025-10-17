import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


plt.style.use('seaborn-v0_8-whitegrid')

# -------------------------
# CadC 活性函数
# -------------------------
def cadc_activity(l, c, pH=6.0):
    """CadC activity calculation based on literature parameters"""
    f_pH = 1 / (1 + 10**((pH - 6.2)/0.5))
    g_l = (l/3.6)**1.1 / (1 + (l/3.6)**1.1)
    h_c = 1 / (1 + (c/0.235)**2.8)
    return f_pH * g_l * h_c

# -------------------------
# 完整 ODE 系统
# -------------------------
def complete_system(t, y, signal_time, pH=6.0):
    """
    Complete ODE system
    y = [m, A_wrapped, A_free, l, c]
    """
    m, A_wrapped, A_free, l, c = y
    
    # CadC activity
    C_active = cadc_activity(l, c, pH)
    
    # Transcription
    C_ratio = 1.21 * C_active
    transcription = 0.0043 * ((1 + C_ratio**2 * 698) / (1 + C_ratio**2))**2
    
    # mRNA kinetics
    dm_dt = transcription - 0.0502 * m
    
    if t < signal_time:
        # Aerobic phase
        dA_wrapped_dt = 0.0042 * m - 0.000398 * A_wrapped
        dA_free_dt = -0.000398 * A_free
    else:
        # Anaerobic phase
        dA_wrapped_dt = 0
        dA_free_dt = 0.0042 * m - 0.000398 * A_free
    
    # Metabolic kinetics
    catalytic_rate = 0.0013 * A_free * l / (26 + l)
    dl_dt = -catalytic_rate
    dc_dt = catalytic_rate
    
    return [dm_dt, dA_wrapped_dt, dA_free_dt, dl_dt, dc_dt]

# -------------------------
# 模拟并绘制 mRNA / 酶 / 产物
# -------------------------
def simulate_and_plot_separate(signal_time=60, simulation_time=180, pH=6.0, l0=10.0):
    y0 = [0.0, 0.0, 0.0, l0, 0.0]
    t_eval = np.linspace(0, simulation_time, 1000)
    
    solution = solve_ivp(
        lambda t, y: complete_system(t, y, signal_time, pH),
        [0, simulation_time],
        y0,
        t_eval=t_eval,
        method='RK45'
    )
    
    t = solution.t
    m, A_wrapped, A_free, l, c = solution.y
    A_total = A_wrapped + A_free
    
    # 1. mRNA + 蛋白质
    plt.figure(figsize=(10, 6))
    plt.plot(t, m, 'b-', linewidth=2, label='mRNA')
    plt.plot(t, A_total, 'r-', linewidth=2, label='Total CadA Protein')
    plt.plot(t, A_wrapped, 'g--', linewidth=2, label='Wrapped Enzyme')
    plt.plot(t, A_free, 'm--', linewidth=2, label='Free Enzyme')
    plt.axvline(x=signal_time, color='k', linestyle=':', alpha=0.7, label='Signal Trigger')
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (a.u.)')
    plt.title('mRNA and Protein Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mRNA_protein_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 代谢物动力学
    plt.figure(figsize=(10, 6))
    plt.plot(t, l, 'orange', linewidth=2, label='Lysine')
    plt.plot(t, c, 'purple', linewidth=2, label='Cadaverine')
    plt.axvline(x=signal_time, color='k', linestyle=':', alpha=0.7, label='Signal Trigger')
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (mM)')
    plt.title('Metabolite Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('metabolite_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Cadaverine 生成速率
    plt.figure(figsize=(10, 6))
    dc_dt = np.gradient(c, t)
    plt.plot(t, dc_dt, 'red', linewidth=2)
    plt.axvline(x=signal_time, color='k', linestyle=':', alpha=0.7, label='Signal Trigger')
    plt.xlabel('Time (min)')
    plt.ylabel('Production Rate (mM/min)')
    plt.title('Cadaverine Production Rate')
    max_rate = np.max(dc_dt)
    max_rate_time = t[np.argmax(dc_dt)]
    plt.annotate(f'Max Rate: {max_rate:.4f} mM/min\nat {max_rate_time:.1f} min', 
                 xy=(max_rate_time, max_rate), xytext=(max_rate_time+20, max_rate*0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=10, ha='left')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('production_rate.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出结果
    print("="*50)
    print("Simulation Results Summary")
    print("="*50)
    print(f"Initial Lysine Concentration: {l0} mM")
    print(f"Final Cadaverine Concentration: {c[-1]:.3f} mM")
    print(f"Conversion Yield: {c[-1]/l0*100:.2f}%")
    print(f"Maximum Production Rate: {max_rate:.4f} mM/min")
    print(f"Time to Maximum Production Rate: {max_rate_time:.1f} min")
    print("\nFigures saved as: mRNA_protein_dynamics.png, metabolite_dynamics.png, production_rate.png")
    
    return solution

# -------------------------
# 只绘制酶动力学
# -------------------------
def plot_enzyme_dynamics_only(signal_time=60, simulation_time=180, pH=6.0, l0=10.0):
    y0 = [0.0, 0.0, 0.0, l0, 0.0]
    t_eval = np.linspace(0, simulation_time, 1000)
    
    solution = solve_ivp(
        lambda t, y: complete_system(t, y, signal_time, pH),
        [0, simulation_time],
        y0,
        t_eval=t_eval,
        method='RK45'
    )
    
    t = solution.t
    A_wrapped, A_free = solution.y[1], solution.y[2]
    A_total = A_wrapped + A_free
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, A_total, 'k-', linewidth=3, label='Total Enzyme')
    plt.plot(t, A_wrapped, 'g-', linewidth=2, label='Wrapped Enzyme')
    plt.plot(t, A_free, 'm-', linewidth=2, label='Free Enzyme')
    plt.axvline(x=signal_time, color='r', linestyle='--', alpha=0.7, label='Signal Trigger')
    plt.xlabel('Time (min)')
    plt.ylabel('Enzyme Concentration (a.u.)')
    plt.title('CadA Enzyme Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'Signal at t = {signal_time} min', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig('enzyme_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("="*50)
    print("Enzyme Dynamics Summary")
    print("="*50)
    print(f"Signal trigger time: {signal_time} min")
    print(f"Maximum total enzyme concentration: {np.max(A_total):.3f} a.u.")
    print(f"Wrapped enzyme at signal trigger: {A_wrapped[np.argmin(np.abs(t - signal_time))]:.3f} a.u.")
    print(f"Free enzyme at signal trigger: {A_free[np.argmin(np.abs(t - signal_time))]:.3f} a.u.")
    print(f"Time to reach 90% of max enzyme: {t[np.where(A_total >= 0.9 * np.max(A_total))[0][0]]:.1f} min")
    
    return solution

# -------------------------
# 参数敏感性分析
# -------------------------
def parameter_sensitivity_analysis_plot(signal_times=[30,60,90,120], initial_lysine=[5,10,15,20]):
    colors = ['blue', 'red', 'green', 'orange']
    t_eval = np.linspace(0,180,1000)
    

    # 图1: 信号时间对产物形成的影响
    plt.figure(figsize=(10,6))
    for i, signal_time in enumerate(signal_times):
        y0 = [0.0,0.0,0.0,10.0,0.0]
        sol = solve_ivp(lambda t,y: complete_system(t,y,signal_time,6.0),
                        [0,180],y0,t_eval=t_eval)
        t = sol.t
        c = sol.y[4]
        plt.plot(t,c,color=colors[i],linewidth=2,label=f'Signal time={signal_time}min')
        plt.axvline(x=signal_time,color=colors[i],linestyle=':',alpha=0.5)
    plt.xlabel('Time (min)')
    plt.ylabel('Cadaverine Concentration (mM)')
    plt.title('Effect of Signal Time on Product Formation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 图2: 初始底物对产物形成的影响
    plt.figure(figsize=(10,6))
    for i, l0 in enumerate(initial_lysine):
        y0 = [0.0,0.0,0.0,l0,0.0]
        sol = solve_ivp(lambda t,y: complete_system(t,y,60,6.0),
                        [0,180],y0,t_eval=t_eval)
        t = sol.t
        c = sol.y[4]
        plt.plot(t,c,color=colors[i],linewidth=2,label=f'Initial Lysine={l0} mM')
    plt.xlabel('Time (min)')
    plt.ylabel('Cadaverine Concentration (mM)')
    plt.title('Effect of Initial Substrate on Product Formation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------------
# 执行模拟
# -------------------------
solution1 = simulate_and_plot_separate(signal_time=60, simulation_time=180, pH=6.0, l0=10.0)
solution2 = plot_enzyme_dynamics_only(signal_time=60, simulation_time=180, pH=6.0, l0=10.0)
parameter_sensitivity_analysis_plot()
