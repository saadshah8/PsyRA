def generate_research_visualizations(performance_metrics):
    """
    Generate comprehensive research-quality visualizations from performance metrics.
    Creates multiple separate plot files instead of a single dashboard.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from scipy import stats
    import matplotlib.patches as mpatches
    
    # Set the style for research-quality plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    
    # Convert metrics to a dataframe for easier processing
    data = []
    for func_name, metrics in performance_metrics.items():
        for metric in metrics:
            data.append({
                "function": func_name,
                "latency": metric["time"],
                "tokens": metric["tokens"],
                "memory": metric["memory"],
                "timestamp": metric["timestamp"],
                "execution_order": len(data) + 1,  # To track sequence
                "stage": metric.get("stage", "unknown")  # Include stage if available
            })
    
    df = pd.DataFrame(data)
    
    if len(df) < 3:
        print("Not enough data for meaningful visualizations.")
        return
    
    # Function colors for consistency across plots
    function_colors = {}
    unique_functions = df['function'].unique()
    color_palette = sns.color_palette("viridis", len(unique_functions))
    for i, func in enumerate(unique_functions):
        function_colors[func] = color_palette[i]
    
    # 1. Performance over time - multi-metric view
    plt.figure(figsize=(16, 12))
    
    # Create a figure with gridspec for separate analysis text area
    fig = plt.figure(figsize=(20, 12))
    # Fix: Ensure width_ratios matches the number of columns (2 in this case)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])  # 4:1 ratio for plot:text
    ax1 = fig.add_subplot(gs[0, 0])  # Main plot area
    ax_text = fig.add_subplot(gs[0, 1])  # Text area
    ax_text.axis('off')  # Turn off axes for text area
    
    # Normalize the metrics for multi-metric comparison
    df_norm = df.copy()
    for metric in ['latency', 'tokens', 'memory']:
        max_val = df_norm[metric].max()
        if max_val > 0:  # Avoid division by zero
            df_norm[f'{metric}_norm'] = df_norm[metric] / max_val
    
    # Plot normalized metrics over execution order with enhanced styling
    for func in unique_functions:
        func_data = df_norm[df_norm['function'] == func]
        ax1.plot(func_data['execution_order'], func_data['latency_norm'], 
                 'o-', color=function_colors[func], alpha=0.7, label=f"{func} (Latency)",
                 linewidth=2, markersize=8)
        ax1.plot(func_data['execution_order'], func_data['memory_norm'], 
                 's--', color=function_colors[func], alpha=0.5, label=f"{func} (Memory)",
                 linewidth=2, markersize=8)

    ax1.set_title('System Performance Metrics Over Conversation Timeline', fontsize=18)
    ax1.set_xlabel('Execution Sequence', fontsize=14)
    ax1.set_ylabel('Normalized Value', fontsize=14)
    ax1.legend(loc='upper center', fontsize=12)
    
    # Add detailed annotation for insight (now in the side panel)
    ax_text.text(0, 0.5, 
             "Insight: This plot shows the relative resource usage patterns throughout the conversation.\n\n"
             "- Parallel lines indicate consistent resource utilization ratios\n\n"
             "- Diverging lines show changing efficiency patterns\n\n"
             "- Spikes identify potential bottlenecks or resource-intensive operations",
             fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig('plot1_performance_over_time.png', dpi=300, bbox_inches='tight')
    print("Saved plot 1 to 'plot1_performance_over_time.png'")
    plt.close()
    
    # 2. Function comparison - horizontal bar chart with detailed performance metrics
    fig = plt.figure(figsize=(20, 12))
    # Fix: Ensure width_ratios matches the number of columns (2 in this case)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])  # 4:1 ratio for plot:text
    ax1 = fig.add_subplot(gs[0, 0])  # Main plot area
    ax_text = fig.add_subplot(gs[0, 1])  # Text area
    ax_text.axis('off')  # Turn off axes for text area
    
    # Calculate detailed stats for each function
    function_means = df.groupby('function').agg({
        'latency': ['mean', 'std', 'max'],
        'tokens': ['mean', 'sum', 'max'],
        'memory': ['mean', 'max']
    }).reset_index()
    
    # Sort by mean latency for better visualization
    function_means = function_means.sort_values(('latency', 'mean'), ascending=True)
    
    # Create horizontal bar chart
    bar_heights = function_means[('latency', 'mean')]
    bar_positions = np.arange(len(function_means))
    bars = ax1.barh(bar_positions, bar_heights, 
                   color=[function_colors[func] for func in function_means['function']], 
                   height=0.5, alpha=0.8)
    
    # Add error bars
    ax1.errorbar(function_means[('latency', 'mean')], bar_positions, 
                xerr=function_means[('latency', 'std')], fmt='none', color='black', capsize=5)
    
    ax1.set_yticks(bar_positions)
    ax1.set_yticklabels(function_means['function'])
    
    # Add values and annotations to each bar
    for i, (bar, func) in enumerate(zip(bars, function_means['function'])):
        # Add main latency value
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{function_means.iloc[i][('latency', 'mean')]:.3f}s", 
                va='center', fontweight='bold')
        
        # Add token count as text to the right
        token_sum = function_means.iloc[i][('tokens', 'sum')]
        token_mean = function_means.iloc[i][('tokens', 'mean')]
        ax1.text(function_means[('latency', 'mean')].max() * 1.2, bar.get_y() + bar.get_height()/2, 
                f"Total: {int(token_sum)} tokens (Avg: {int(token_mean)}/call)", 
                va='center', color='darkblue', fontsize=10)
    
    ax1.set_title('Performance Comparison Across System Components', fontsize=18)
    ax1.set_xlabel('Latency (seconds)', fontsize=14)
    
    # Add performance analysis annotation to the side panel
    if len(function_means) >= 2:
        # Find the fastest and slowest functions
        fastest_func = function_means.iloc[0]['function']
        slowest_func = function_means.iloc[-1]['function']
        fastest_time = function_means.iloc[0][('latency', 'mean')]
        slowest_time = function_means.iloc[-1][('latency', 'mean')]
        speedup = slowest_time / fastest_time if fastest_time > 0 else 0
        
        ax_text.text(0, 0.5, 
                f"Performance Analysis:\n\n"
                f"• {fastest_func} is the most efficient component ({fastest_time:.3f}s avg.)\n\n"
                f"• {slowest_func} is the most time-consuming ({slowest_time:.3f}s avg.)\n\n"
                f"• Performance ratio: {speedup:.1f}x difference between fastest and slowest components\n\n"
                f"• The distribution suggests {'balanced system design' if speedup < 3 else 'optimization opportunities'}",
                fontsize=12, va='center', bbox=dict(facecolor='lavender', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig('plot2_function_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved plot 2 to 'plot2_function_comparison.png'")
    plt.close()
    
    # 3. Enhanced token efficiency analysis
    fig = plt.figure(figsize=(20, 12))
    # Fix: Ensure width_ratios matches the number of columns (2 in this case)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])  # 4:1 ratio for plot:text
    ax1 = fig.add_subplot(gs[0, 0])  # Main plot area
    ax_text = fig.add_subplot(gs[0, 1])  # Text area
    ax_text.axis('off')  # Turn off axes for text area
    
    # Calculate token processing rate
    df['tokens_per_second'] = df['tokens'] / df['latency'].clip(lower=0.001)
    
    # Box plot of token processing efficiency
    sns.boxplot(x='function', y='tokens_per_second', hue='function', 
                data=df, palette=function_colors, legend=False, ax=ax1)
    
    ax1.set_title('Token Processing Efficiency by Component', fontsize=18)
    ax1.set_xlabel('')  # X labels are function names
    ax1.set_ylabel('Tokens per Second', fontsize=14)
    
    # Fix the ticklabel warning by explicitly setting ticks first
    ax1.set_xticks(range(len(unique_functions)))
    ax1.set_xticklabels([func for func in unique_functions], rotation=45, ha='right')
    
    # Add mean values as text with more detailed information
    for i, func in enumerate(unique_functions):
        func_data = df[df['function'] == func]
        mean_tps = func_data['tokens_per_second'].mean()
        median_tps = func_data['tokens_per_second'].median()
        max_tps = func_data['tokens_per_second'].max()
        
        ax1.text(i, df['tokens_per_second'].min(), 
                f"Mean: {mean_tps:.1f}\nMedian: {median_tps:.1f}\nMax: {max_tps:.1f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Add efficiency analysis to the side panel
    highest_efficiency_func = df.groupby('function')['tokens_per_second'].mean().idxmax()
    highest_efficiency = df.groupby('function')['tokens_per_second'].mean().max()
    
    ax_text.text(0, 0.5, 
            f"Efficiency Analysis:\n\n"
            f"• {highest_efficiency_func} shows the highest token processing efficiency\n\n"
            f"• Average efficiency: {highest_efficiency:.1f} tokens/second\n\n"
            f"• Higher values indicate better computational efficiency\n\n"
            f"• Large variance suggests inconsistent performance",
            fontsize=12, va='center',
            bbox=dict(facecolor='honeydew', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig('plot3_token_efficiency.png', dpi=300, bbox_inches='tight')
    print("Saved plot 3 to 'plot3_token_efficiency.png'")
    plt.close()
    
    # 4. Memory vs. Latency scatter plot with regression analysis
    fig = plt.figure(figsize=(20, 12))
    # Fix: Ensure width_ratios matches the number of columns (2 in this case)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])  # 4:1 ratio for plot:text
    ax1 = fig.add_subplot(gs[0, 0])  # Main plot area
    ax_text = fig.add_subplot(gs[0, 1])  # Text area
    ax_text.axis('off')  # Turn off axes for text area
    
    # Calculate token and memory efficiency metrics
    df['tokens_per_mb'] = df['tokens'] / df['memory'].clip(lower=0.001)
    
    # Size points by token count for added dimension
    sizes = df['tokens'] / df['tokens'].max() * 200 + 50
    
    # Plot with enhanced styling and annotations
    for func in unique_functions:
        func_data = df[df['function'] == func]
        scatter = ax1.scatter(func_data['latency'], func_data['memory'], 
                    color=function_colors[func], alpha=0.7, label=func, 
                    s=sizes[func_data.index], edgecolors='white', linewidth=0.5)
        
        # Add linear regression line with confidence interval if enough data points
        if len(func_data) >= 3:
            x = func_data['latency']
            y = func_data['memory']
            
            # Calculate regression
            m, b = np.polyfit(x, y, 1)
            
            # Plot regression line
            x_range = np.linspace(x.min(), x.max(), 100)
            ax1.plot(x_range, m*x_range + b, color=function_colors[func], 
                     linestyle='--', alpha=0.7, linewidth=2)
            
            # Calculate R-squared
            r_squared = np.corrcoef(x, y)[0, 1]**2
            
            # Add regression equation and R² text
            if len(x) > 3:  # Only add if we have enough data points
                middle_x = x.min() + (x.max() - x.min()) * 0.6
                ax1.text(middle_x, m*middle_x + b + 0.1, 
                        f"R²={r_squared:.2f}", color=function_colors[func], 
                        fontweight='bold', fontsize=10)
    
    ax1.set_title('Resource Utilization: Memory vs. Latency Analysis', fontsize=18)
    ax1.set_xlabel('Latency (seconds)', fontsize=14)
    ax1.set_ylabel('Memory Usage (MB)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Add size legend for token count
    legend_sizes = [df['tokens'].min(), df['tokens'].mean(), df['tokens'].max()]
    legend_handles = []
    for size in legend_sizes:
        size_scaled = size / df['tokens'].max() * 200 + 50
        handle = mpatches.Circle((0, 0), radius=np.sqrt(size_scaled/np.pi), 
                                 color='gray', alpha=0.5)
        legend_handles.append(handle)
    
    size_legend = ax1.legend(legend_handles, 
                             [f"{int(s)} tokens" for s in legend_sizes],
                             title="Token Count", fontsize=9,
                             loc='lower right', frameon=True, framealpha=0.8)
    
    # Add the original legend back after adding the size legend
    ax1.add_artist(ax1.legend(loc='upper left', fontsize=10))
    
    # Calculate overall correlation coefficient and add detailed analysis to the side panel
    overall_corr = df['memory'].corr(df['latency'])
    ax_text.text(0, 0.5, 
             f"Resource Correlation Analysis:\n\n"
             f"• Overall correlation between memory and latency: {overall_corr:.2f}\n\n"
             f"• {'Strong' if abs(overall_corr) > 0.7 else 'Moderate' if abs(overall_corr) > 0.3 else 'Weak'} "
             f"{'positive' if overall_corr > 0 else 'negative'} correlation\n\n"
             f"• Larger points indicate higher token count\n\n"
             f"• Steeper slopes indicate higher memory cost per unit time\n\n"
             f"• Clustering reveals consistent resource patterns",
             fontsize=12, va='center',
             bbox=dict(facecolor='mistyrose', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig('plot4_memory_vs_latency.png', dpi=300, bbox_inches='tight')
    print("Saved plot 4 to 'plot4_memory_vs_latency.png'")
    plt.close()
    
    # 5. Performance distribution histogram with density curve
    fig = plt.figure(figsize=(20, 12))
    # Fix: Ensure width_ratios matches the number of columns (2 in this case)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])  # 4:1 ratio for plot:text
    ax1 = fig.add_subplot(gs[0, 0])  # Main plot area
    ax_text = fig.add_subplot(gs[0, 1])  # Text area
    ax_text.axis('off')  # Turn off axes for text area
    
    # Create histograms for latency distributions with KDE
    for func in unique_functions:
        func_data = df[df['function'] == func]
        if len(func_data) >= 3:  # Need at least 3 points for a meaningful distribution
            sns.histplot(func_data['latency'], bins=max(5, min(10, len(func_data))), 
                         alpha=0.4, label=func, color=function_colors[func], 
                         kde=True, stat='density', ax=ax1)
            
            # Calculate and annotate important distribution statistics
            mean_val = func_data['latency'].mean()
            median_val = func_data['latency'].median()
            
            # Add vertical lines for mean and median
            ax1.axvline(mean_val, color=function_colors[func], linestyle='-', alpha=0.7)
            ax1.axvline(median_val, color=function_colors[func], linestyle=':', alpha=0.7)
    
    ax1.set_title('Component Response Time Distribution Analysis', fontsize=18)
    ax1.set_xlabel('Latency (seconds)', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    
    # Add distribution analysis text to the side panel
    ax_text.text(0, 0.5, 
             "Distribution Analysis:\n\n"
             "• Solid lines show mean values\n\n"
             "• Dotted lines show median values\n\n"
             "• Wider distributions indicate higher variability\n\n"
             "• Left-skewed distributions suggest optimized performance\n\n"
             "• Multi-modal distributions may indicate context-dependent performance",
             fontsize=12, va='center',
             bbox=dict(facecolor='aliceblue', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig('plot5_performance_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved plot 5 to 'plot5_performance_distribution.png'")
    plt.close()
    
    # Conversation stage analysis plots (if stage data is available)
    if ('stage' in df.columns and len(df['stage'].unique()) > 1) or any('stage' in metric for func, metrics in performance_metrics.items() for metric in metrics):
        # Extract stage information if not explicitly provided
        if 'stage' not in df.columns or df['stage'].nunique() <= 1:
            df['stage'] = df['function'].apply(lambda x: x.split('_')[-1] if '_' in x else 'unknown')
        
        # Skip if still not enough stage variety
        if df['stage'].nunique() <= 1:
            print("Not enough distinct stages for stage analysis.")
        else:
            # Use a different color palette for stages
            stage_colors = dict(zip(df['stage'].unique(), sns.color_palette("Set2", df['stage'].nunique())))
            
            # 6. Enhanced performance by stage with multi-metric view
            fig = plt.figure(figsize=(20, 12))
            # Fix: Ensure width_ratios matches the number of columns (2 in this case)
            gs = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])  # 4:1 ratio for plot:text
            ax1 = fig.add_subplot(gs[0, 0])  # Main plot area
            ax_text = fig.add_subplot(gs[0, 1])  # Text area
            ax_text.axis('off')  # Turn off axes for text area
            
            # Calculate stage-wise statistics
            stage_performance = df.groupby('stage').agg({
                'latency': ['mean', 'std', 'median', 'count'],
                'tokens': ['mean', 'sum', 'max'],
                'memory': ['mean', 'max']
            }).reset_index()
            
            # Create main bar chart for latency with error bars
            x = np.arange(len(stage_performance))
            width = 0.35
            
            latency_bars = ax1.bar(x - width/2, stage_performance[('latency', 'mean')], width, 
                    yerr=stage_performance[('latency', 'std')], label='Avg Latency (s)',
                    color=[stage_colors[stage] for stage in stage_performance['stage']], 
                    capsize=5, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add token bars on secondary axis
            ax_right = ax1.twinx()
            token_bars = ax_right.bar(x + width/2, stage_performance[('tokens', 'mean')], width, 
                         label='Avg Tokens per Response', color='lightcoral', alpha=0.6,
                         edgecolor='darkred', linewidth=1, hatch='//')
            
            # Improve axis labels and title
            ax1.set_xlabel('Conversation Stage', fontsize=14)
            ax1.set_ylabel('Latency (seconds)', fontsize=14, color='black')
            ax_right.set_ylabel('Token Count', fontsize=14, color='darkred')
            ax1.set_title('Performance Profile Across Conversation Stages', fontsize=18)
            
            # Set tick positions and labels
            ax1.set_xticks(x)
            ax1.set_xticklabels(stage_performance['stage'], fontsize=12)
            
            # Add count annotations to bars
            for i, bar in enumerate(latency_bars):
                count = stage_performance.iloc[i][('latency', 'count')]
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"n={count}", ha='center', va='bottom', fontsize=10)
            
            # Add token annotations
            for i, bar in enumerate(token_bars):
                token_sum = stage_performance.iloc[i][('tokens', 'sum')]
                ax_right.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f"{int(token_sum)} total", ha='center', va='bottom', 
                               fontsize=10, color='darkred')
            
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax_right.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
            
            # Add stage analysis insights to the side panel
            if len(stage_performance) >= 2:
                # Find the most and least efficient stages
                fastest_stage = stage_performance.iloc[stage_performance[('latency', 'mean')].idxmin()]['stage']
                slowest_stage = stage_performance.iloc[stage_performance[('latency', 'mean')].idxmax()]['stage']
                highest_token_stage = stage_performance.iloc[stage_performance[('tokens', 'mean')].idxmax()]['stage']
                
                # Add analysis text
                ax_text.text(0, 0.5, 
                        f"Conversation Stage Analysis:\n\n"
                        f"• {fastest_stage} stage has the lowest latency\n\n"
                        f"• {slowest_stage} stage has the highest latency\n\n"
                        f"• {highest_token_stage} stage generates the most tokens\n\n"
                        f"• Performance varies {stage_performance[('latency', 'mean')].max() / stage_performance[('latency', 'mean')].min():.1f}x across stages\n\n"
                        f"• {'Token count correlates with latency' if df['tokens'].corr(df['latency']) > 0.5 else 'Token count does not strongly predict latency'}",
                        fontsize=12, va='center',
                        bbox=dict(facecolor='lightgreen', alpha=0.7, boxstyle='round,pad=0.5'))
            
            plt.tight_layout()
            plt.savefig('plot6_stage_performance.png', dpi=300, bbox_inches='tight')
            print("Saved plot 6 to 'plot6_stage_performance.png'")
            plt.close()
            
            # 7. Enhanced latency trend by stage with smoothed trend lines
            fig = plt.figure(figsize=(20, 12))
            # Fix: Ensure width_ratios matches the number of columns (2 in this case)
            gs = GridSpec(1, 2, figure=fig, width_ratios=[4, 1])  # 4:1 ratio for plot:text
            ax1 = fig.add_subplot(gs[0, 0])  # Main plot area
            ax_text = fig.add_subplot(gs[0, 1])  # Text area
            ax_text.axis('off')  # Turn off axes for text area
            
            for stage in df['stage'].unique():
                stage_data = df[df['stage'] == stage]
                if len(stage_data) >= 3:  # Need at least 3 points for a meaningful trend
                    # Plot raw data points
                    ax1.scatter(stage_data['execution_order'], stage_data['latency'], 
                               label=stage, alpha=0.7, color=stage_colors[stage], s=80,
                               edgecolors='white', linewidth=0.5)
                    
                    # Add smoothed trend line if we have enough points
                    if len(stage_data) >= 5:
                        # Simple moving average for smoothing
                        window_size = min(3, len(stage_data) // 2)
                        if window_size >= 2:
                            x = stage_data['execution_order'].values
                            y = stage_data['latency'].values
                            y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                            x_smooth = x[window_size-1:]
                            
                            # Plot smoothed line
                            ax1.plot(x_smooth, y_smooth, '-', color=stage_colors[stage], 
                                    alpha=0.8, linewidth=2)
            
            ax1.set_title('Latency Evolution Throughout Conversation', fontsize=18)
            ax1.set_xlabel('Interaction Sequence', fontsize=14)
            ax1.set_ylabel('Response Time (seconds)', fontsize=14)
            ax1.legend(title='Conversation Stage', fontsize=10)
            
            # Add trend analysis to the side panel
            if df['execution_order'].nunique() >= 3:
                # Calculate overall latency trend
                x = df['execution_order']
                y = df['latency']
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trend_direction = "increasing" if slope > 0 else "decreasing"
                trend_significance = "significant" if p_value < 0.05 else "not significant"
                
                ax_text.text(0, 0.5, 
                        f"Temporal Trend Analysis:\n\n"
                        f"• Overall response time is {trend_direction} over the conversation\n\n"
                        f"• Trend is statistically {trend_significance} (p={p_value:.3f})\n\n"
                        f"• Slope: {slope:.4f} seconds per interaction\n\n"
                        f"• R² value: {r_value**2:.3f}\n\n"
                        f"• {'System shows performance degradation over time' if slope > 0 else 'System maintains or improves efficiency over time'}",
                        fontsize=12, va='center',
                        bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.5'))
            
            plt.tight_layout()
            plt.savefig('plot7_latency_trend.png', dpi=300, bbox_inches='tight')
            print("Saved plot 7 to 'plot7_latency_trend.png'")
            plt.close()
            
            # 8. Enhanced memory usage by stage with statistical annotations
            plt.figure(figsize=(16, 12))
            
            # Box plot with fixed hue parameter
            sns.boxplot(x='stage', y='memory', hue='stage', data=df, palette=stage_colors, 
                        legend=False)
            
            plt.title('Memory Footprint by Conversation Stage', fontsize=18)
            plt.xlabel('Conversation Stage', fontsize=14)
            plt.ylabel('Memory Consumption (MB)', fontsize=14)
            
            # Calculate and add statistical annotations
            stage_memory_stats = df.groupby('stage')['memory'].agg(['mean', 'median', 'std', 'max']).reset_index()
            
            # Find stage with highest memory usage
            max_memory_stage = stage_memory_stats.iloc[stage_memory_stats['max'].idxmax()]['stage']
            max_memory_value = stage_memory_stats['max'].max()
            
            # Add annotations for statistical insights
            plt.text(0.01, 0.99, 
                    f"Memory Usage Analysis:\n"
                    f"• Peak memory consumption: {max_memory_value:.2f} MB during {max_memory_stage} stage\n"
                    f"• Memory utilization varies significantly across stages\n"
                    f"• {'Higher memory usage stages correlate with higher token generation' if df['memory'].corr(df['tokens']) > 0.5 else 'Memory usage does not directly correlate with token generation'}\n"
                    f"• Memory profiling suggests {'optimization opportunities' if max_memory_value > 100 else 'efficient resource management'}",
                    transform=plt.gca().transAxes, fontsize=12, va='top',
                    bbox=dict(facecolor='lavender', alpha=0.8, boxstyle='round,pad=0.5'))

            plt.tight_layout()
            plt.savefig('plot8_memory_by_stage.png', dpi=300, bbox_inches='tight')
            print("Saved plot 8 to 'plot8_memory_by_stage.png'")
            plt.close()

            
    print("All visualizations generated as separate PNG files.")
    return True