import os
import numpy as np
import astropy.io.fits as fits
from xshooter_func import SpectrumProcessor as SP
from xshooter_func import piecewise_spectrum, normalize_spectrum, remove_absorbed_regions, galactic_deredden
from scipy.interpolate import interp1d
import datetime
import warnings
from astropy.table import Table
import time
warnings.filterwarnings("ignore")

sp = SP()


WLQ_list = sp.WLQ_LIST

# 读取标准光谱
with fits.open(os.path.join(sp.molecfit_data_dir, '0210-0823', f'0210-0823_full_spectrum.fits')) as hdu_norm:
    wave_norm =  hdu_norm[1].data['WAVE']
    flux_norm = hdu_norm[1].data['FLUX']
    err_norm = hdu_norm[1].data['ERR']
    if 'REDSHIFT' in hdu_norm[0].header:
        redshift_norm = hdu_norm[0].header['REDSHIFT']
    else:
        redshift_norm = hdu_norm[0].header['DR16Z']

    wave_norm = wave_norm / (1 + redshift_norm)
    flux_norm = flux_norm * (1 + redshift_norm)
    err_norm = err_norm * (1 + redshift_norm)



band_config = {
    'UVB': {
        'wave_common': np.arange(800, 2400, 0.08),
        'reference_windows': None,  # 根据实际定义
        'pixel_size': 5,
        'wave_cutoff': (sp.WAVE_CUTOFFS['UVB'], sp.WAVE_CUTOFFS['VIS'])
    },
    'VIS': {
        'wave_common': np.arange(1500, 4200, 0.25),
        'reference_windows': [(1970., 2400.), (2480., 2675.), (2925., 3400.)],
        'pixel_size': 5,
        'wave_cutoff': (sp.WAVE_CUTOFFS['VIS'], sp.WAVE_CUTOFFS['NIR'][0])
    },
    'NIR': {
        'wave_common': np.arange(2700, 8400, 0.25),
        'reference_windows': [(4000., 4050.), (4200., 4230.), (5100., 5535.), (6005., 6035.)],
        'pixel_size': 5,
        'wave_cutoff': (sp.WAVE_CUTOFFS['NIR'][0], sp.WAVE_CUTOFFS['NIR'][1])
    }
}


def process_band(wave_obs, flux_obs, err_obs, redshift, wave_norm, flux_norm,
                 wave_common, reference_windows, pixel_size=5):
    # 红移校正
    wave_rest = wave_obs / (1 + redshift)
    flux_rest = flux_obs * (1 + redshift)
    err_rest = err_obs * (1 + redshift)

    # 归一化
    flux_rest_norm = normalize_spectrum(wave_norm, flux_norm, wave_rest, flux_rest,
                                        reference_windows=reference_windows)
    scale = flux_rest_norm / flux_rest
    err_rest_norm = err_rest * scale

    # 分段处理
    wave_pieces, flux_pieces, err_pieces = piecewise_spectrum(
        wave_rest, flux_rest_norm, err_rest_norm, pixel_size=pixel_size
    )
    ranges = [(piece[0], piece[-1]) for piece in wave_pieces]

    # 创建掩膜
    mask = np.zeros_like(wave_common, dtype=bool)
    for start, end in ranges:
        idx = (wave_common >= start) & (wave_common <= end)
        mask[idx] = True

    # 插值
    f_flux = interp1d(wave_rest, flux_rest_norm, kind='linear',
                      bounds_error=False, fill_value=np.nan)
    f_err = interp1d(wave_rest, err_rest_norm, kind='linear',
                     bounds_error=False, fill_value=np.nan)

    flux_interp = f_flux(wave_common)
    err_interp = f_err(wave_common)

    # 应用掩膜
    flux_interp[~mask] = np.nan
    err_interp[~mask] = np.nan

    return flux_interp, err_interp


# 初始化存储结构
bands = ['UVB', 'VIS', 'NIR']
stacks = {band: {'flux': [], 'err': []} for band in bands}

for source in WLQ_list:
    source_path = os.path.join(sp.molecfit_data_dir, source, f'{source}_full_spectrum.fits')

    with fits.open(source_path) as hdul:
        wave_obs = hdul[1].data['WAVE']
        flux_obs = hdul[1].data['FLUX']
        err_obs = hdul[1].data['ERR']
        RA = hdul[0].header['RA']
        DEC = hdul[0].header['DEC']
        redshift = np.round(hdul[0].header['redshift'], 4)

    # # 堆叠高红移的源时跳过低红移的源，只在特殊情况下使用
    # if redshift < 2:
    #     continue

    print(f'processing {source} ..., redshift={redshift}')
    # 去除吸收线
    wave_obs, flux_obs, err_obs = remove_absorbed_regions(wave_obs, flux_obs, err_obs, mask_regions=sp.mask_regions[source])
    # 去红化
    flux_obs, err_obs = galactic_deredden(wave_obs, flux_obs, err_obs, RA, DEC)

    # 分波段处理
    for band in bands:
        cfg = band_config[band]
        # 提取当前波段数据
        idx = (wave_obs >= cfg['wave_cutoff'][0]) & (wave_obs <= cfg['wave_cutoff'][1])
        wave_band = wave_obs[idx]
        flux_band = flux_obs[idx]
        err_band = err_obs[idx]

        # 处理波段
        flux_interp, err_interp = process_band(
            wave_band, flux_band, err_band, redshift,
            wave_norm, flux_norm,
            cfg['wave_common'],
            cfg['reference_windows'],
            cfg['pixel_size']
        )

        stacks[band]['flux'].append(flux_interp)
        stacks[band]['err'].append(err_interp)


# 计算中值/均值复合光谱  —— 已替换为 Bootstrap (含测量误差注入) 方法
def bootstrap_median_with_noise(flux_stack, err_stack, B=1000, seed=42, return_percentiles=True):
    """
    flux_stack: ndarray (N_spectra, N_wave) with NaN for masked pixels
    err_stack:  ndarray (N_spectra, N_wave) per-pixel 1-sigma measurement errors (NaN where masked)
    B: number of bootstrap iterations
    返回:
      median_spec: median of original stack (N_wave,)
      sigma_boot: std of bootstrap medians (N_wave,)
      lo16, hi84: 16th and 84th percentiles of bootstrap medians (N_wave,) if return_percentiles
    """
    rng = np.random.default_rng(seed)
    flux_stack = np.array(flux_stack)
    err_stack = np.array(err_stack)
    N, M = flux_stack.shape
    if N == 0:
        if return_percentiles:
            return (np.full(M, np.nan), np.full(M, np.nan),
                    np.full(M, np.nan), np.full(M, np.nan))
        else:
            return (np.full(M, np.nan), np.full(M, np.nan))

    medians_b = np.empty((B, M), dtype=float)
    for b in range(B):
        idx = rng.integers(0, N, size=N)  # 有放回抽样
        sampled_flux = flux_stack[idx, :].copy()    # shape (N, M)
        sampled_err  = err_stack[idx, :].copy()     # shape (N, M)

        # 对被 mask 的像素 err 为 NaN -> 将 NaN 转为 0.0，使得生成 noise 时这些位置噪声为0
        sampled_err_nonan = np.nan_to_num(sampled_err, 0.0)

        # 生成噪声并加入（对被 mask 的像素不会影响，因为 flux 为 NaN）
        noise = rng.normal(loc=0.0, scale=1.0, size=sampled_flux.shape) * sampled_err_nonan
        sampled_flux_noisy = sampled_flux + noise

        # 计算该次抽样的中值（按列，忽略 NaN）
        medians_b[b, :] = np.nanmedian(sampled_flux_noisy, axis=0)

    median_spec = np.nanmedian(flux_stack, axis=0)
    sigma_boot = np.nanstd(medians_b, axis=0, ddof=1)   # bootstrap sigma (包含样本变动 + 测量误差)
    if return_percentiles:
        lo16 = np.nanpercentile(medians_b, 16, axis=0)
        hi84 = np.nanpercentile(medians_b, 84, axis=0)
        return median_spec, sigma_boot, lo16, hi84
    else:
        return median_spec, sigma_boot

# 参数：根据计算资源和精度调整 B（B=1000 是常用折中）
B_BOOT = 1000
SEED = 20250819

comp_specs = {}
t0 = time.time()
for band in bands:
    flux_stack = np.array(stacks[band]['flux'])   # shape (Nspec, Nwave)
    err_stack  = np.array(stacks[band]['err'])    # same shape

    # 每个波长点的有效谱数
    n_obs = np.sum(~np.isnan(flux_stack), axis=0)

    # 原始中值谱（不注入噪声）
    flux_median = np.nanmedian(flux_stack, axis=0)

    # Bootstrap + noise injection -> sigma_total, 16/84 percentiles
    if flux_stack.size == 0:
        sigma_boot = np.full_like(flux_median, np.nan)
        lo16 = np.full_like(flux_median, np.nan)
        hi84 = np.full_like(flux_median, np.nan)
    else:
        flux_median, sigma_boot, lo16, hi84 = bootstrap_median_with_noise(
            flux_stack, err_stack, B=B_BOOT, seed=SEED, return_percentiles=True
        )

    comp_specs[band] = {
        'wave': band_config[band]['wave_common'],
        'flux_median': flux_median,
        'err_boot': sigma_boot,
        'err_lo16': lo16,
        'err_hi84': hi84,
        'n_obs': n_obs
    }
print(f"bootstrap done (B={B_BOOT}), elapsed {time.time()-t0:.1f}s for bands {bands}")

# 保存到FITS（用 bootstrap sigma 作为 ERR，并同时保存 16/84 百分位）
primary_hdu = fits.PrimaryHDU(header=fits.Header({'Author': 'Cheng Xiaoqiang'}))
primary_hdu.header['DATE'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
primary_hdu.header['NOTE'] = 'Median composite spectrum (bootstrap errors include sample scatter + measurement noise)'
primary_hdu.header['BOOTB'] = (B_BOOT, 'Bootstrap iterations used')
primary_hdu.header['EMAIL'] = '15107937394@163.com'
hdul = fits.HDUList([primary_hdu])

for band in bands:
    data = comp_specs[band]
    valid = ~np.isnan(data['flux_median'])
    table = Table({
        'WAVE': data['wave'][valid],
        'FLUX': data['flux_median'][valid],
        'ERR': data['err_boot'][valid],
        'ERR16': data['err_lo16'][valid],
        'ERR84': data['err_hi84'][valid],
        'SNR': np.divide(data['flux_median'][valid], data['err_boot'][valid], out=np.full_like(data['flux_median'][valid], np.nan), where=data['err_boot'][valid]!=0),
        'COUNTS': data['n_obs'][valid]
    })

    hdu = fits.BinTableHDU(table)
    hdu.header['EXTNAME'] = band
    hdu.header['WAVE'] = 'Wavelength (Angstrom)'
    hdu.header['FLUX'] = 'Flux (10^-17 erg/s/cm^2/Angstrom)'
    hdu.header['ERR'] = 'Bootstrap sigma (includes sample scatter + measurement error)'
    hdu.header['ERR16'] = '16th percentile of bootstrap medians'
    hdu.header['ERR84'] = '84th percentile of bootstrap medians'
    hdu.header['SNR'] = 'Flux / ERR'
    hdu.header['COUNTS'] = 'Number of input spectra contributing per lambda'
    hdul.append(hdu)

hdul.writeto('median_comp_spec_bootstrap.fits', overwrite=True)
# hdul.writeto('high_z_comp_spec.fits', overwrite=True)
