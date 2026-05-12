import numpy as np
import os
import csv
import time as time_module
import warnings
import logging
import types
import contextlib
import multiprocessing as mp
from itertools import count

# PyRK 라이브러리 임포트
from pyrk.utilities.ur import units
from pyrk.timer import Timer
from pyrk.th_component import THComponent
from pyrk.materials.material import Material
from pyrk.materials.liquid_material import LiquidMaterial
from pyrk.density_model import DensityModel
from pyrk.reactivity_insertion import ReactivityInsertion
from pyrk.db import database
from pyrk.inp import sim_info
from pyrk.driver import solve

# ---------------------------------------------------------
# 0. 환경 설정
# ---------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

OUTPUT_CSV = "nuscale_training_dataset_10k.csv"
TARGET_SAMPLES = 10000
NUM_WORKERS = mp.cpu_count() - 2

# ✅ 완전 균일 랜덤 샘플링 (Uniform Distribution)
RHO_MIN = -0.020
RHO_MAX = -0.0001
DUR_MIN = 3.0
DUR_MAX = 40.0

# ✅ 재현 가능한 랜덤 시드
RANDOM_SEED = 42

print("=" * 60)
print(f"🚀 NuScale 학습 데이터 생성기 (멀티프로세싱)")
print(f"⚡ {NUM_WORKERS}개 워커 병렬 실행 (예상 {NUM_WORKERS}배 속도)")
print("=" * 60)
print(f"🎲 완전 균일 랜덤 샘플링 (Uniform Distribution)")
print(f"   • 반응도: {RHO_MIN * 1e5:.0f} ~ {RHO_MAX * 1e5:.0f} pcm")
print(f"   • 제어 시간: {DUR_MIN:.1f} ~ {DUR_MAX:.1f}초")
print(f"   • 랜덤 시드: {RANDOM_SEED} (재현 가능)")
print(f"\n   예상 분포 (균일 샘플링 시):")
print(f"   - 정상 운전 (~75%): -32 ~ -585 pcm")
print(f"   - 위험 영역 (~20%): -585 ~ -975 pcm")
print(f"   - Shutdown (~5%): -975 ~ -1170 pcm")
print("-" * 60)


# ---------------------------------------------------------
# 1. 시뮬레이션 함수
# ---------------------------------------------------------
def run_single_scenario(args):
    """
    단일 시뮬레이션 실행

    ✅ 완전 균일 랜덤 샘플링 (Uniform Distribution)
    - 반응도: RHO_MIN ~ RHO_MAX 사이 균일 분포
    - 제어 시간: DUR_MIN ~ DUR_MAX 사이 균일 분포
    - 각 태스크는 고유 시드로 재현 가능

    ⚠️ CSV 저장은 메인 프로세스에서만 수행 (race condition 방지)
    """
    task_id, seed = args

    # ✅ 태스크별 고유 시드 설정 (재현성 보장)
    np.random.seed(seed)

    # ✅ 완전 균일 랜덤 샘플링
    rho_target = np.random.uniform(RHO_MIN, RHO_MAX)
    duration_target = np.random.uniform(DUR_MIN, DUR_MAX)

    ti = Timer(t0=0 * units.s, tf=300.0 * units.s, dt=1.0 * units.s)
    power_tot = 250e6 * units.watt

    H_FLOW_EQUILIBRIUM = 29492.3829
    h_fuel = 8500.0 * units.watt / (units.meter ** 2 * units.kelvin)
    A = 752.0 * units.meter ** 2
    T_cool_0 = 580.0 * units.kelvin
    T_fuel_0 = T_cool_0 + power_tot / (h_fuel * A)

    alpha_f = -3.0e-5 * units.delta_k / units.kelvin
    alpha_c = -20.0e-5 * units.delta_k / units.kelvin

    class UO2(Material):
        def __init__(self):
            super().__init__(name="uo2", k=3.0 * units.watt / units.meter / units.kelvin,
                             cp=310.0 * units.joule / units.kg / units.kelvin,
                             dm=DensityModel(a=10970.0 * units.kg / units.meter ** 3, model="constant"))

    class Water(LiquidMaterial):
        def __init__(self):
            super().__init__(name="water", k=0.56 * units.watt / units.meter / units.kelvin,
                             cp=5600.0 * units.joule / units.kg / units.kelvin,
                             dm=DensityModel(a=720.0 * units.kg / units.meter ** 3, model="constant"),
                             mu=9.0e-5 * units.pascal * units.second)

    fuel = THComponent(name="fuel", mat=UO2(), vol=3.0 * units.meter ** 3, T0=T_fuel_0,
                       alpha_temp=alpha_f, timer=ti, heatgen=True, power_tot=power_tot)
    cool = THComponent(name="cool", mat=Water(), vol=15.0 * units.meter ** 3, T0=T_cool_0,
                       alpha_temp=alpha_c, timer=ti)
    inlet = THComponent(name="inlet", mat=Water(), vol=15.0 * units.meter ** 3, T0=553.0 * units.kelvin,
                        alpha_temp=0.0 * units.delta_k / units.kelvin, timer=ti)

    fuel.add_convection("cool", h=h_fuel, area=A)
    cool.add_convection("fuel", h=h_fuel, area=A)
    cool.add_convection("inlet", h=H_FLOW_EQUILIBRIUM * units.watt / units.meter ** 2 / units.kelvin, area=A)

    class CustomRampReactivity(ReactivityInsertion):
        def __init__(self, timer, start_time, dur, total_rho):
            super().__init__(timer)
            self.start_time = start_time
            self.end_time = start_time + dur
            self.total_rho = total_rho
            self.slope = total_rho / dur

        def reactivity(self, t_idx=None):
            if t_idx is None:
                t_idx = 0
            current_time = (self.timer.t0.magnitude + t_idx * self.timer.dt.magnitude)

            if current_time < self.start_time:
                return 0.0 * units.delta_k
            elif self.start_time <= current_time < self.end_time:
                dt = current_time - self.start_time
                return (self.slope * dt) * units.delta_k
            else:
                return self.total_rho * units.delta_k

    rho_ext = CustomRampReactivity(timer=ti, start_time=100.0,
                                   dur=duration_target, total_rho=rho_target)

    db_name = f"tmp_{task_id}_{os.getpid()}.h5"
    db = database.Database(db_name)
    si = sim_info.SimInfo(timer=ti, components=[fuel, cool, inlet],
                          iso="u235", e="thermal", n_precursors=6, n_decay=11,
                          rho_ext=rho_ext, feedback=True, db=db)

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        infile = types.SimpleNamespace(kappa=0.0, nsteps=10000)
        sol = solve(si=si, y=si.y, infile=infile)

    db.close_db()
    if os.path.exists(db_name):
        try:
            os.remove(db_name)
        except:
            pass

    sol = np.array(sol)
    power = sol[:, 0] * 250
    T_fuel_arr = sol[:, -3] - 273.15
    T_cool_arr = sol[:, -2] - 273.15

    times = np.linspace(0, 300.0, len(power))
    idx_100 = np.argmin(np.abs(times - 100.0))
    idx_end = -1

    result = {
        'id': task_id,
        'input_reactivity': rho_target,
        'input_duration': duration_target,
        'power_initial': power[idx_100],
        'T_fuel_initial': T_fuel_arr[idx_100],
        'T_cool_initial': T_cool_arr[idx_100],
        'power_final': power[idx_end],
        'T_fuel_final': T_fuel_arr[idx_end],
        'T_cool_final': T_cool_arr[idx_end],
        'power_min': np.min(power[idx_100:])
    }

    # 검증
    if (result['T_fuel_final'] < -200 or
            result['T_cool_final'] < -200 or
            np.isnan(result['power_final']) or
            np.isinf(result['power_final'])):
        return None  # 실패시 None 반환

    return result


# ---------------------------------------------------------
# 2. 무한 태스크 생성기
# ---------------------------------------------------------
def task_generator():
    """
    무한 태스크 생성 (필요한 만큼만 소비됨)
    각 태스크마다 재현 가능한 고유 시드 부여
    """
    for i in count():
        # ✅ 재현 가능: RANDOM_SEED + i * 소수(137)
        seed = RANDOM_SEED + i * 137
        yield (i, seed)


# ---------------------------------------------------------
# 3. 메인 실행
# ---------------------------------------------------------
if __name__ == '__main__':
    print("데이터 생성 시작...\n")

    headers = ['id', 'input_reactivity', 'input_duration',
               'power_initial', 'T_fuel_initial', 'T_cool_initial',
               'power_final', 'T_fuel_final', 'T_cool_final',
               'power_min']

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    start_time = time_module.time()
    success_count = 0
    failed_count = 0

    # ✅ Pool 생성 및 스트리밍 처리
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # imap_unordered: 완료되는 대로 결과 반환 (순서 무관)
        for result in pool.imap_unordered(run_single_scenario, task_generator(), chunksize=1):
            if result is not None:
                # ✅ 메인 프로세스에서만 CSV 저장 (race condition 방지)
                with open(OUTPUT_CSV, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerow(result)

                success_count += 1

                # ✅ 진행 상황 (더 자주 출력)
                if True:
                    elapsed = time_module.time() - start_time
                    rate = success_count / elapsed if elapsed > 0 else 0
                    remain = (TARGET_SAMPLES - success_count) / rate if rate > 0 else 0

                    status = "SD" if result['power_final'] < 10.0 else "OP"

                    print(f"[{success_count:5d}/{TARGET_SAMPLES}] "
                          f"P={result['power_final']:5.1f}MW ({status}) | "
                          f"{rate:.1f}샘플/초 | ETA: {remain / 60:.1f}분")

                # 목표 달성시 즉시 종료
                if success_count >= TARGET_SAMPLES:
                    pool.terminate()
                    break
            else:
                failed_count += 1

    total_time = time_module.time() - start_time

    # ✅ 실제 샘플 분포 계산
    print("\n생성된 데이터 분석 중...")
    with open(OUTPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    normal = sum(1 for row in data if float(row['input_reactivity']) > -0.009)
    risky = sum(1 for row in data if -0.015 <= float(row['input_reactivity']) <= -0.009)
    shutdown = sum(1 for row in data if float(row['input_reactivity']) < -0.015)

    print("\n" + "=" * 60)
    print(f"🎉 완료!")
    print(f"파일: {OUTPUT_CSV}")
    print(f"시간: {total_time / 60:.1f}분 ({total_time:.0f}초)")
    print(f"성공: {success_count}개")
    print(f"실패: {failed_count}개 (자동 건너뜀)")
    print(f"속도: {success_count / total_time:.2f} 샘플/초")
    print(f"\n📊 실제 샘플 분포:")
    print(f"   • 정상 운전: {normal:,}개 ({normal / success_count * 100:.1f}%)")
    print(f"   • 위험 영역: {risky:,}개 ({risky / success_count * 100:.1f}%)")
    print(f"   • Shutdown: {shutdown:,}개 ({shutdown / success_count * 100:.1f}%)")
    print("=" * 60)