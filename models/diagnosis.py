import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def linear_regression_diagnostics(model_pipeline, x_train,xtest, y_train,ytest):

    if isinstance(model_pipeline, TransformedTargetRegressor):
        inner_pipeline = model_pipeline.regressor_
    else:
        inner_pipeline = model_pipeline

    preprocessor = inner_pipeline[:-1]

    x_train_processed = inner_pipeline[:-1].transform(x_train)
    xtest_processed = inner_pipeline[:-1].transform(xtest)

    
    try:
        feature_names = inner_pipeline[:-1].get_feature_names_out()
    except:
        feature_names = [f"x{i}" for i in range(x_train_processed.shape[1])]

    x_train_processed = pd.DataFrame(x_train_processed, columns=feature_names)
    
    # Predictions
    y_pred_train = model_pipeline.predict(x_train)
    y_pred_test = model_pipeline.predict(xtest)

    
    # Performance Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(ytest, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(ytest, y_pred_test)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    rmse_test = root_mean_squared_error(ytest, y_pred_test)
    
    print(f"R² Score (Train): {r2_train:.4f}")
    print(f"R² Score (Test): {r2_test:.4f}")
    print(f"MSE (Train): {mse_train:.4f}")
    print(f"MSE (Test): {mse_test:.4f}")
    print(f"RMSE (Train): {rmse_train:.4f}")
    print(f"RMSE (Test): {rmse_test:.4f}")
    
    # Residuals
    residuals = y_train - y_pred_train
    
    print("=" * 55)
    print("LINEAR REGRESSION — FULL ASSUMPTION DIAGNOSTICS")
    print("=" * 55)

    # ─────────────────────────────────────────────────────────────────
    # ASSUMPTION 1 — LINEARITY
    # Visual only: scatter each feature vs y
    # ─────────────────────────────────────────────────────────────────
    print("\n[1] LINEARITY → Check scatter plots (x vs y)")
    print("    No single number — use your eyes on the plots below.")

    # ─────────────────────────────────────────────────────────────────
    # ASSUMPTION 2 — INDEPENDENCE (Durbin-Watson)
    # ─────────────────────────────────────────────────────────────────
    dw = durbin_watson(residuals)
    print(f"\n[2] INDEPENDENCE — Durbin-Watson")
    print(f"    DW statistic : {dw:.4f}")
    if 1.5 <= dw <= 2.5:
        print("    Result       : PASSED ✓ (between 1.5 and 2.5)")
    elif dw < 1.5:
        print("    Result       : FAILED ✗ — positive autocorrelation")
    else:
        print("    Result       : FAILED ✗ — negative autocorrelation")

    # ─────────────────────────────────────────────────────────────────
    # ASSUMPTION 3 — NORMALITY (Shapiro-Wilk)
    # ─────────────────────────────────────────────────────────────────
    stat_sw, p_sw = stats.shapiro(residuals)
    print(f"\n[3] NORMALITY — Shapiro-Wilk")
    print(f"    Statistic : {stat_sw:.4f}")
    print(f"    p-value   : {p_sw:.4f}")
    if p_sw > 0.05:
        print("    Result    : PASSED ✓ (p > 0.05)")
    else:
        print("    Result    : FAILED ✗ (p < 0.05)")

    # ─────────────────────────────────────────────────────────────────
    # ASSUMPTION 4 — EQUAL VARIANCE (Breusch-Pagan)
    # ─────────────────────────────────────────────────────────────────
    X_with_const  = sm.add_constant(x_train_processed)
    bp_stat, p_bp, _, _ = het_breuschpagan(residuals, X_with_const)
    print(f"\n[4] EQUAL VARIANCE — Breusch-Pagan")
    print(f"    Statistic : {bp_stat:.4f}")
    print(f"    p-value   : {p_bp:.4f}")
    if p_bp > 0.05:
        print("    Result    : PASSED ✓ (p > 0.05 — homoscedastic)")
    else:
        print("    Result    : FAILED ✗ (p < 0.05 — heteroscedastic)")

    # ─────────────────────────────────────────────────────────────────
    # ASSUMPTION 5 — MULTICOLLINEARITY (VIF)
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[5] MULTICOLLINEARITY — VIF scores")
    vif_data = pd.DataFrame()
    vif_data['Feature'] = x_train_processed.columns
    vif_data['VIF']     = [variance_inflation_factor(x_train_processed.values, i)
                            for i in range(x_train_processed.shape[1])]
    print(vif_data.to_string(index=False))
    for _, row in vif_data.iterrows():
        if row['VIF'] > 10:
            print(f"    WARNING ✗ : {row['Feature']} has VIF={row['VIF']:.1f} — serious problem")
        elif row['VIF'] > 5:
            print(f"    CAUTION   : {row['Feature']} has VIF={row['VIF']:.1f} — moderate concern")
        else:
            print(f"    OK ✓      : {row['Feature']} has VIF={row['VIF']:.1f}")

    # ─────────────────────────────────────────────────────────────────
    # PLOTS — All visuals in one figure
    # ─────────────────────────────────────────────────────────────────

    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Assumption Diagnostics', fontsize=14)
    # 2. Independence: residuals vs row order
    axes[0,0].plot(residuals.values, color='purple', alpha=0.6, linewidth=0.8)
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_title(f'Independence — residuals over time (DW={dw:.2f})')
    axes[0,0].set_xlabel('Row order')
    axes[0,0].set_ylabel('Residual')

    # 3. Normality: Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title(f'Normality — Q-Q plot (p={p_sw:.3f})')

    # 3. Homoscedasticity (FIXED → scatter instead of kde)
    axes[1,0].scatter(y_pred_train, residuals, alpha=0.5)
    axes[1,0].axhline(0, color='red', linestyle='--')
    axes[1,0].set_title(f'Residuals vs Fitted (p={p_bp:.3f})')
    axes[1,0].set_xlabel('Fitted values')
    axes[1,0].set_ylabel('Residuals')

    # 4. Equal variance: residuals vs fitted
    sns.kdeplot(
        x=y_pred_train,
        y=residuals,
        fill=True,
        ax=axes[1,1]
    )
    axes[1,1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1,1].set_title(f'Equal Variance — residuals vs fitted (p={p_bp:.3f})')
    axes[1,1].set_xlabel('Fitted values')
    axes[1,1].set_ylabel('Residuals')

    # 5. Multicollinearity: correlation heatmap
    # numeric_cols = x_train.select_dtypes(include='number')

    # corr = numeric_cols.corr()
    corr = x_train_processed.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm',
                center=0, ax=axes[2,1], square=True,fmt='.2f',cbar_kws={"shrink": 0.7}, vmin=-1, vmax=1)
    axes[2,1].set_title('Multicollinearity — correlation heatmap')

    # 6. VIF bar chart
    colors = ['green' if v < 5 else 'orange' if v < 10 else 'red'
            for v in vif_data['VIF']]
    axes[2,0].bar(vif_data['Feature'], vif_data['VIF'], color=colors)
    axes[2,0].tick_params(axis='x', rotation=90)
    axes[2,0].axhline(5,  color='orange', linestyle='--', linewidth=1.2, label='VIF=5')
    axes[2,0].axhline(10, color='red',    linestyle='--', linewidth=1.2, label='VIF=10')
    axes[2,0].set_title('Multicollinearity — VIF scores')
    axes[2,0].set_ylabel('VIF'); axes[1,1].legend()

    plt.tight_layout()
    plt.show()

    # for col in x_train_processed.columns[:10]:  # limit to first 4
    #     plt.figure()
    #     plt.scatter(x_train_processed[col], y_train, alpha=0.3)
    #     plt.title(f"{col} vs Target")
    #     plt.xlabel(col)
    #     plt.ylabel("Target")
    #     plt.show()