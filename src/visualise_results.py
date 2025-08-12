import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Check available fonts and use the best one
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]

# Choose best available font
if 'DejaVu Sans' in available_fonts:
    font_family = 'DejaVu Sans'
elif 'Arial' in available_fonts:
    font_family = 'Arial'
elif 'Helvetica' in available_fonts:
    font_family = 'Helvetica'
else:
    font_family = 'sans-serif'

print(f"Using font: {font_family}")

# Modern DeepMind-inspired color palette
COLORS = {
    'primary': '#1f4e79',      # Deep Blue
    'secondary': '#4285f4',    # Google Blue  
    'accent': '#ea4335',       # Google Red
    'neutral': '#5f6368',      # Medium Gray
    'dark': '#202124',         # Almost Black
    'light': '#e8eaed',        # Visible Light Gray
    'grid': '#dadce0',         # Grid Gray
    'background': '#f8f9fa'    # Background
}

# Modern styling configuration
plt.style.use('default')  # Start fresh
plt.rcParams.update({
    'font.family': font_family,
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'axes.linewidth': 0.8,     # Visible axes
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.bottom': True,
    'ytick.left': True,
    'grid.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': COLORS['background']
})

def load_and_clean_data():
    """Load and clean the benchmark data"""
    csv_path = 'results/benchmark_results_generalized_20250810_001708.csv'
    df = pd.read_csv(csv_path)
    
    cols = [
        'model_name', 'model_type', 'provider', 'batch_size',
        'avg_latency_ms', 'p95_latency_ms', 'p99_latency_ms',
        'peak_memory_mb', 'avg_memory_mb', 'model_size_mb'
    ]
    df = df[cols]
    
    # Modern model name cleaning
    def clean_name(name):
        # Remove common prefixes/suffixes
        replacements = {
            'financial-sentiment': '', 'financial-classifier': '',
            'all-MiniLM-L6-v2-': 'MiniLM ', 'distilbert-': 'DistilBERT ',
            'finbert-tone-': 'FinBERT ', 'mobilebert-uncased-': 'MobileBERT ',
            'tinybert-': 'TinyBERT ', 'SmolLM2-360M-Instruct-': 'SmolLM2 ',
            '-fine-tuned': ' (FT)', '-pruned': ' (Pruned)'
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # Clean formatting
        name = name.replace('-', ' ').replace('_', ' ')
        name = ' '.join(name.split())  # Remove extra spaces
        return name.strip()
    
    df['model_name'] = df['model_name'].apply(clean_name)
    
    # Filter and clean providers
    df_plot = df[
        (df['batch_size'] == 1) & 
        (df['provider'].str.contains('CPU|CoreML', case=False, regex=True))
    ].copy()
    
    df_plot['provider'] = df_plot['provider'].map({
        'CPUExecutionProvider': 'CPU',
        'CoreMLExecutionProvider': 'CoreML'
    })
    
    return df_plot

def create_modern_barplot(data, x_col, y_col, hue_col, title, ylabel, filename, figsize=(12, 7)):
    """Create a modern, clean bar plot"""
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Get unique providers and create color mapping
    providers = data[hue_col].unique()
    if len(providers) == 1:
        colors = [COLORS['primary']]
    else:
        colors = [COLORS['primary'], COLORS['secondary']]
    
    # Create grouped bar plot manually for better control
    models = data[x_col].unique()
    x_pos = np.arange(len(models))
    bar_width = 0.6 if len(providers) == 1 else 0.4  # Wider bars, less spacing
    
    for i, provider in enumerate(providers):
        provider_data = data[data[hue_col] == provider]
        values = [provider_data[provider_data[x_col] == model][y_col].iloc[0] 
                 if len(provider_data[provider_data[x_col] == model]) > 0 else 0 
                 for model in models]
        
        offset = (i - len(providers)/2 + 0.5) * bar_width if len(providers) > 1 else 0
        bars = ax.bar(x_pos + offset, 
                     values, bar_width, 
                     label=provider, 
                     color=colors[i], 
                     alpha=0.85,
                     edgecolor='white',
                     linewidth=1)
        
        # Add subtle value labels
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'{value:.0f}', ha='center', va='bottom', 
                       fontsize=10, color=COLORS['dark'], alpha=0.8, fontweight='semibold')
    
    # Modern styling
    ax.set_title(title, fontsize=20, fontweight='bold', color=COLORS['dark'], pad=25)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='semibold', color=COLORS['dark'])
    ax.set_xlabel('')  # Remove x-label for cleaner look
    
    # Clean x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=0, ha='center', color=COLORS['dark'])
    
    # Visible y-axis with proper styling
    ax.tick_params(axis='y', colors=COLORS['neutral'], length=4)
    ax.tick_params(axis='x', colors=COLORS['neutral'], length=4)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    
    # Visible grid
    ax.grid(axis='y', color=COLORS['grid'], linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Style the visible spines
    ax.spines['left'].set_color(COLORS['neutral'])
    ax.spines['bottom'].set_color(COLORS['neutral'])
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Modern legend with better visibility
    if len(providers) > 1:
        legend = ax.legend(frameon=True, loc='upper right', 
                          bbox_to_anchor=(0.98, 0.98),
                          fancybox=True, shadow=False,
                          framealpha=0.95, edgecolor=COLORS['neutral'])
        legend.get_frame().set_facecolor('white')
        for text in legend.get_texts():
            text.set_color(COLORS['dark'])
            text.set_fontweight('semibold')
    
    # Keep top and right spines hidden for modern look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Clean layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.9)
    
    # High-quality save
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    return fig

def create_modern_comparison_plot(data, figsize=(14, 10)):
    """Create a modern 3-panel comparison plot"""
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Define metrics
    metrics = [
        ('avg_latency_ms', 'Inference Latency', 'ms'),
        ('peak_memory_mb', 'Peak Memory', 'MB'),
        ('model_size_mb', 'Model Size', 'MB')
    ]
    
    for idx, (metric, title, unit) in enumerate(metrics, 1):
        ax = plt.subplot(2, 2, idx)
        
        # Get data for this metric
        models = data['model_name'].unique()
        providers = data['provider'].unique()
        
        x_pos = np.arange(len(models))
        bar_width = 0.6 if len(providers) == 1 else 0.35  # Tighter spacing
        
        colors = [COLORS['primary'], COLORS['secondary']] if len(providers) > 1 else [COLORS['primary']]
        
        for i, provider in enumerate(providers):
            provider_data = data[data['provider'] == provider]
            values = [provider_data[provider_data['model_name'] == model][metric].iloc[0] 
                     if len(provider_data[provider_data['model_name'] == model]) > 0 else 0 
                     for model in models]
            
            offset = (i - len(providers)/2 + 0.5) * bar_width if len(providers) > 1 else 0
            bars = ax.bar(x_pos + offset, 
                         values, bar_width, 
                         label=provider, 
                         color=colors[i], 
                         alpha=0.85,
                         edgecolor='white',
                         linewidth=0.8)
        
        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold', color=COLORS['dark'], pad=15)
        ax.set_ylabel(unit, fontsize=12, color=COLORS['dark'], fontweight='semibold')
        
        if idx == 3:  # Only show x-labels on bottom plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right', color=COLORS['dark'])
        else:
            ax.set_xticks([])
        
        # Clean styling with visible elements
        ax.grid(axis='y', color=COLORS['grid'], linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor(COLORS['background'])
        
        # Style spines
        ax.spines['left'].set_color(COLORS['neutral'])
        ax.spines['bottom'].set_color(COLORS['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(left=True, bottom=True, colors=COLORS['neutral'])
        
        if idx == 1 and len(providers) > 1:  # Add legend to first plot
            legend = ax.legend(frameon=True, loc='upper right',
                             framealpha=0.95, edgecolor=COLORS['neutral'])
            legend.get_frame().set_facecolor('white')
            for text in legend.get_texts():
                text.set_color(COLORS['dark'])
                text.set_fontweight('semibold')
    
    # Add overall title
    fig.suptitle('Model Performance Benchmark', fontsize=22, fontweight='bold', 
                 color=COLORS['dark'], y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.3)
    
    plt.savefig('modern_comparison_overview.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    return fig

# Main execution
print("Loading data and creating modern visualizations...")

# Load data
df_plot = load_and_clean_data()

print(f"Creating plots for {len(df_plot)} model configurations...")
print(f"Models: {list(df_plot['model_name'].unique())}")
print(f"Providers: {list(df_plot['provider'].unique())}")

# Create individual modern plots
create_modern_barplot(
    data=df_plot,
    x_col='model_name',
    y_col='avg_latency_ms', 
    hue_col='provider',
    title='Inference Latency',
    ylabel='Latency (ms)',
    filename='modern_latency.png'
)

create_modern_barplot(
    data=df_plot,
    x_col='model_name',
    y_col='peak_memory_mb',
    hue_col='provider', 
    title='Peak Memory Usage',
    ylabel='Memory (MB)',
    filename='modern_memory.png'
)

create_modern_barplot(
    data=df_plot,
    x_col='model_name',
    y_col='model_size_mb',
    hue_col='provider',
    title='Model File Size', 
    ylabel='Size (MB)',
    filename='modern_size.png'
)

# Create comparison overview
create_modern_comparison_plot(df_plot)

print("\n✨ Modern visualizations created!")
print("Generated files:")
print("• modern_latency.png")
print("• modern_memory.png") 
print("• modern_size.png")
print("• modern_comparison_overview.png")