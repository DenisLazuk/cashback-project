"""
Cashback Optimization System
============================
This module helps optimize cashback by selecting the best bank card categories
based on cashback ratios, limits, and bank-specific constraints.

Author: Generated from CASHBACK_app.ipynb
Date: 2026-02-01
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Iterable, Tuple, List, Optional


# ==============================================================================
# BANK DATA DEFINITIONS
# ==============================================================================

# Tinkoff Bank (tbank)
# Условия: 5000 суммарный размер кешбека в месяц по всем категориям
# с подпиской Pro иначе 3000
tbank = {
    ('tbank', 'на все', 0.01, 5000, 'all', 1, 4, 0),
    ('tbank', 'искусство', 0.05, 5000, 'all', 1, 4, 0),
    ('tbank', 'образование', 0.05, 5000, 'all', 1, 4, 0),
    ('tbank', 'досуг', 0.05, 5000, 'all', 1, 4, 0),
    # каршеринг removed — prefer жд билеты for T-Bank
    ('tbank', 'книги', 0.05, 5000, 'all', 1, 4, 0),
    ('tbank', 'жд билеты', 0.05, 5000, 'all', 1, 4, 0),
}

# MTS Bank
# mts = {
#     ('mts', 'продукты', 0.07, 1000, 'one', 1, 5, 0),
#     ('mts', 'цветы', 0.07, 1000, 'one', 1, 5, 0),
#     ('mts', 'фастфуд и рестораны', 0.12, 1000, 'one', 1, 5, 0),
#     ('mts', 'на все', 0.07, 1000, 'one', 1, 5, 0),
#     ('mts', 'досуг', 0.2, 1000, 'one', 1, 5, 0),
#     ('mts', 'дом и ремонт', 0.05, 1000, 'one', 1, 5, 0),
#     ('mts', 'такси', 0.05, 1000, 'all', 1, 5, 0),
#     ('mts', 'маркетплейсы', 0.1, 1000, 'all', 1, 5, 0)
# }

# Domrf Bank
# Условия: 1000 суммарный размер кешбека в месяц по всем категориям
dom = {
    ('dom', 'спорттовары', 0.05, 1000, 'all', 1, 4, 0),
    ('dom', 'подарки', 0.05, 1000, 'all', 1, 4, 0),
    ('dom', 'бытовые услуги', 0.05, 1000, 'all', 1, 4, 0),
    ('dom', 'книги', 0.05, 1000, 'all', 1, 4, 0)
}

# PSB Bank
# Условия: сумма покупок от 5000 в месяц. status=1 = include in selection
psb = {
    ('psb', 'салоны красоты', 0.05, 3000, 'all', 1, 3, 0),
    ('psb', 'такси и каршеринг', 0.05, 3000, 'all', 1, 3, 0),
    ('psb', 'досуг', 0.01, 3000, 'all', 1, 3, 1),
    ('psb', 'книги', 0.05, 3000, 'all', 1, 3, 0),
    ('psb', 'транспорт', 0.05, 3000, 'all', 1, 3, 0),
    ('psb', 'жд билеты', 0.05, 3000, 'all', 1, 3, 0),
    ('psb', 'спорттовары', 0.03, 3000, 'all', 1, 3, 0),
    ('psb', 'аптеки', 0.03, 3000, 'all', 1, 3, 0),
    ('psb', 'одежда', 0.03, 3000, 'all', 1, 3, 0),
    ('psb', 'косметика', 0.03, 3000, 'all', 1, 3, 0),
    ('psb', 'бытовая техника', 0.02, 3000, 'all', 1, 3, 0),
    ('psb', 'дом и ремонт', 0.02, 3000, 'all', 1, 3, 0),
}

# Sberbank
# Условия: 5000 суммарный размер кешбека в месяц по всем категориям
sber = {
    ('sber', 'на все', 0.01, 5000, 'all', 1, 3, 0),
    ('sber', 'медицинские услуги', 0.02, 5000, 'all', 1, 3, 0),
    ('sber', 'парфюмерия и косметика', 0.05, 5000, 'all', 1, 3, 0),
    ('sber', 'питомцы', 0.05, 5000, 'all', 1, 3, 0),
    # такси и каршеринг removed — prefer OTP for this category
    ('sber', 'транспорт', 0.10, 5000, 'all', 1, 3, 0),
    ('sber', 'досуг', 0.05, 5000, 'all', 1, 3, 0),
}

# Alfa Bank
# Условия: 5000 суммарный размер кешбека в месяц по всем категориям
alpha = {
    ('alpha', 'на все', 0.01, 5000, 'all', 1, 3, 0),
    ('alpha', 'спорттовары', 0.05, 5000, 'all', 1, 3, 0),
    ('alpha', 'цифровые товары', 0.05, 5000, 'all', 1, 3, 0),
    ('alpha', 'аптеки', 0.03, 5000, 'all', 1, 3, 0),
    ('alpha', 'дикси доставка', 0.20, 5000, 'all', 1, 3, 0),
}

# OTP Bank
# Условия: 3000 суммарный размер кешбека в месяц по всем категориям
otp = {
    ('otp', 'на все', 0.01, 3000, 'all', 1, 4, 0),
    ('otp', 'коммуналка', 0.03, 3000, 'all', 1, 4, 0),
    ('otp', 'цветы', 0.10, 3000, 'all', 1, 4, 0),
    ('otp', 'аптеки', 0.01, 3000, 'all', 1, 4, 0),
    ('otp', 'медицинские услуги', 0.02, 3000, 'all', 1, 4, 0),
    ('otp', 'супермаркеты', 0.02, 3000, 'all', 1, 4, 0),
    ('otp', 'кафе и рестораны', 0.05, 3000, 'all', 1, 4, 0),
    ('otp', 'искусство', 0.10, 3000, 'all', 1, 4, 0),
    ('otp', 'образование', 0.05, 3000, 'all', 1, 4, 0),
    ('otp', 'одежда и обувь', 0.05, 3000, 'all', 1, 4, 0),
    ('otp', 'такси и каршеринг', 0.05, 3000, 'all', 1, 4, 0),
}

# VTB Bank
# Условия: сумма покупок от 5000 в месяц
vtb = {
    ('vtb', 'дом и ремонт', 0.04, 1000, 'all', 1, 3, 0),
    ('vtb', 'аптеки', 0.04, 1000, 'all', 1, 3, 0),
    ('vtb', 'ювелирные изделия', 0.15, 1000, 'all', 1, 3, 0),
    ('vtb', 'детские товары', 0.04, 1000, 'all', 1, 3, 0),
    ('vtb', 'цветы', 0.08, 1000, 'all', 1, 3, 0),
    ('vtb', 'здоровье', 0.04, 1000, 'all', 1, 3, 0),
}

# Column names for reference
columns = ['name', 'category', 'ratio', 'm_limit', 'limit_type', 'status', 'total_cats', 'bonus']

# Dictionary of all banks
banks_dict = {
    'tbank': tbank,
    # 'mts': mts,
    'dom': dom,
    'psb': psb,
    'sber': sber,
    'alpha': alpha,
    'otp': otp,
    'vtb': vtb
}

# Category priority: common value from top to bottom (на все = super category, then products, etc.)
# Selection fills these first, then unique categories, then best-ratio fill.
PRIORITY_CATEGORIES = [
    'на все',
    'продукты',
    'аптеки',
    'такси и каршеринг',
    'жд билеты',
    'фастфуд',
    'кафе и рестораны',
    'досуг',
]

# Pre-selected (bank, category) — fixed by user, cannot be changed by the algorithm.
# Example: [('tbank', 'на все'), ('alpha', 'аптеки')]
LOCKED_SELECTIONS: List[Tuple[str, str]] = [
    ('tbank', 'на все'),
    ('tbank', 'досуг'),
    ('tbank', 'подарки'),
    ('tbank', 'рестораны'),
    ('vtb', 'продукты'),
    ('vtb', 'цветы'),
    ('vtb', 'образование'),
    ('otp', 'одежда'),
    ('otp', 'продукты'),
    ('otp', 'такси и каршеринг'),
    ('otp', 'коммуналка')
    ]

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def _banks_to_candidates(banks_dict: Dict[str, Iterable[Tuple]]) -> pd.DataFrame:
    """
    Flatten banks_dict into a candidate DataFrame.
    
    Tuple formats accepted:
      - (bank, category, ratio, m_limit, status)
      - (bank, category, ratio, m_limit, status, total_cats)
      - (bank, category, ratio, m_limit, status, total_cats, bonus)
      - (bank, category, ratio, m_limit, limit_type, status, total_cats, bonus)
    
    Parameters
    ----------
    banks_dict : Dict[str, Iterable[Tuple]]
        Dictionary mapping bank names to their category tuples
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['bank', 'category', 'ratio', 'm_limit', 'limit_type', 'status', 
         'total_cats', 'bonus', 'limit_used']
    """
    rows = []
    for bank_name, items in banks_dict.items():
        for t in items:
            lt = len(t)
            if lt == 5:
                bank, category, ratio, m_limit, status = t
                limit_type, total_cats, bonus = 'all', 1, 0.0
            elif lt == 6:
                bank, category, ratio, m_limit, status, total_cats = t
                limit_type, bonus = 'all', 0.0
            elif lt == 7:
                bank, category, ratio, m_limit, status, total_cats, bonus = t
                limit_type = 'all'
            elif lt == 8:  # Current data format
                bank, category, ratio, m_limit, limit_type, status, total_cats, bonus = t
            else:
                raise ValueError(f"Each tuple must have 5-8 elements, got {lt}")

            rows.append({
                'bank': str(bank).strip(),
                'category': str(category).strip(),
                'ratio': ratio,
                'm_limit': m_limit,
                'limit_type': limit_type,
                'status': status,
                'total_cats': total_cats,
                'bonus': bonus,
                'limit_used': 0.0
            })

    df = pd.DataFrame(rows)
    
    # Normalize data types
    df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce').fillna(0.0).astype(float)
    df['m_limit'] = pd.to_numeric(df['m_limit'], errors='coerce').fillna(0.0).astype(float)
    df['status'] = pd.to_numeric(df['status'], errors='coerce').fillna(1).astype(int)
    df['total_cats'] = pd.to_numeric(df['total_cats'], errors='coerce').fillna(1).astype(int)
    df['limit_used'] = pd.to_numeric(df['limit_used'], errors='coerce').fillna(0.0).astype(float)
    df['bonus'] = pd.to_numeric(df['bonus'], errors='coerce').fillna(0).astype(float)
    
    # Set column order
    df = df[['bank', 'category', 'ratio', 'm_limit', 'limit_type', 'status', 
             'total_cats', 'bonus', 'limit_used']]
    return df


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a safe copy and ensure expected columns and types exist.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate and normalize
    
    Returns
    -------
    pd.DataFrame
        Validated and normalized DataFrame
    """
    d = df.copy()
    
    # Normalize possible limit column name
    if 'm_limit' not in d.columns and 'amount_RUR' in d.columns:
        d = d.rename(columns={'amount_RUR': 'm_limit'})

    # Ensure essential columns
    for c in ['bank', 'category', 'ratio', 'm_limit', 'status', 'total_cats']:
        if c not in d.columns:
            raise ValueError(f"Input DataFrame must contain column: {c}")

    # Normalize types
    d['ratio'] = pd.to_numeric(d['ratio'], errors='coerce').fillna(0.0).astype(float)
    d['m_limit'] = pd.to_numeric(d['m_limit'], errors='coerce').fillna(0.0).astype(float)

    # limit_used optional — if absent, set 0
    if 'limit_used' not in d.columns:
        d['limit_used'] = 0.0
    else:
        d['limit_used'] = pd.to_numeric(d['limit_used'], errors='coerce').fillna(0.0).astype(float)

    d['status'] = pd.to_numeric(d['status'], errors='coerce').fillna(1).astype(int)
    d['total_cats'] = pd.to_numeric(d['total_cats'], errors='coerce').fillna(1).astype(int)

    # bonus optional
    if 'bonus' not in d.columns:
        d['bonus'] = 0.0
    else:
        d['bonus'] = pd.to_numeric(d['bonus'], errors='coerce').fillna(0.0).astype(float)

    # Strip category/bank strings
    d['category'] = d['category'].astype(str).str.strip()
    d['bank'] = d['bank'].astype(str).str.strip()
    return d


def select_best_global(
    banks_dict: Dict[str, Iterable[Tuple]],
    low_priority_banks: Optional[List[str]] = None,
    priority_categories: Optional[List[str]] = None,
    locked_selections: Optional[Iterable[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Three-phase selection:
    0) Add all locked (bank, category) first — given by user, cannot be changed.
    1) For each priority category, pick best by ratio across banks (MTS last).
    2) Fill empty slots with unique categories (only one bank has it).
    3) Fill remaining slots with any category not yet selected for that bank, best ratio first.
    """
    if low_priority_banks is None:
        low_priority_banks = []
    low_priority_banks = [str(b).strip().lower() for b in low_priority_banks]
    if priority_categories is None:
        priority_categories = list(PRIORITY_CATEGORIES)
    if locked_selections is None:
        locked_selections = []

    candidates = _banks_to_candidates(banks_dict)
    candidates = candidates[candidates['status'] != 0].copy()

    candidates['_bank_priority'] = candidates['bank'].str.strip().str.lower().apply(
        lambda b: 1 if b in low_priority_banks else 0
    )

    bank_capacity = candidates[['bank', 'total_cats']].drop_duplicates().set_index('bank')['total_cats'].to_dict()
    for b in list(bank_capacity.keys()):
        bank_capacity.setdefault(b, int(bank_capacity[b]))
    bank_used = {b: 0 for b in bank_capacity.keys()}
    selected_rows: List[dict] = []

    def has_capacity(bank: str) -> bool:
        return bank_used.get(bank, 0) < bank_capacity.get(bank, 0)

    selected_pairs: set = set()  # (bank, category) already chosen — never add twice

    def add_row(row_series, locked: bool = False) -> None:
        d = row_series.to_dict()
        key = (d['bank'], d['category'])
        if key in selected_pairs:
            return
        selected_pairs.add(key)
        d['locked'] = 1 if locked else 0
        selected_rows.append(d)
        bank_used[d['bank']] = bank_used.get(d['bank'], 0) + 1

    # --- Phase 0: locked (pre-selected) categories — fixed by user ---
    for bank_key, cat_key in locked_selections:
        bank_key = str(bank_key).strip()
        cat_key = str(cat_key).strip()
        match = candidates[(candidates['bank'] == bank_key) & (candidates['category'] == cat_key)]
        if match.empty or not has_capacity(bank_key):
            continue
        add_row(match.iloc[0], locked=True)

    # --- Phase 1: priority categories in order, pick best by ratio (MTS last) ---
    for cat in priority_categories:
        mask = (candidates['category'] == cat)
        available = candidates[mask]
        if available.empty:
            continue
        # Only banks with capacity
        available = available[available['bank'].apply(has_capacity)]
        if available.empty:
            continue
        best = available.sort_values(
            ['_bank_priority', 'ratio', 'm_limit', 'bonus'],
            ascending=[True, False, False, False]
        ).iloc[0]
        add_row(best)

    # --- Phase 2: unique categories (only one bank has this category), fill slots ---
    cat_to_banks = candidates.groupby('category')['bank'].apply(lambda x: list(x.unique())).to_dict()
    unique_cats = [c for c, banks in cat_to_banks.items() if len(banks) == 1]
    unique_rows = candidates[candidates['category'].isin(unique_cats)]
    unique_rows = unique_rows.sort_values(
        ['_bank_priority', 'ratio', 'm_limit', 'bonus'],
        ascending=[True, False, False, False]
    )
    for idx, row in unique_rows.iterrows():
        if not has_capacity(row['bank']):
            continue
        add_row(row)

    # --- Phase 3: fill remaining slots with best ratio (any category not yet selected for this bank) ---
    selected_per_bank: Dict[str, set] = {}
    for r in selected_rows:
        b = r['bank']
        selected_per_bank.setdefault(b, set()).add(r['category'])

    remaining = candidates.copy()
    while True:
        fill = remaining[
            remaining.apply(lambda r: has_capacity(r['bank']) and r['category'] not in selected_per_bank.get(r['bank'], set()), axis=1)
        ]
        if fill.empty:
            break
        best = fill.sort_values(
            ['_bank_priority', 'ratio', 'm_limit', 'bonus'],
            ascending=[True, False, False, False]
        ).iloc[0]
        add_row(best)
        selected_per_bank.setdefault(best['bank'], set()).add(best['category'])
        remaining = remaining.drop(index=best.name)

    selected_df = pd.DataFrame(selected_rows)
    if '_bank_priority' in selected_df.columns:
        selected_df = selected_df.drop(columns=['_bank_priority'], errors='ignore')

    if not selected_df.empty:
        # Sort by priority order then category then ratio for readable output
        def _cat_order(c):
            try:
                return priority_categories.index(c)
            except ValueError:
                return len(priority_categories)
        selected_df['_order'] = selected_df['category'].apply(_cat_order)
        selected_df = selected_df.sort_values(by=['_order', 'category', 'ratio'], ascending=[True, True, False]).reset_index(drop=True)
        selected_df = selected_df.drop(columns=['_order'], errors='ignore')
        selected_df.insert(0, 'rank', range(1, len(selected_df) + 1))

    return selected_df


def use_limit(df: pd.DataFrame, bank: str, category: str, amount: float) -> pd.DataFrame:
    """
    Increases limit_used for a category and disables it (status=0)
    when limit_used >= m_limit.
    
    Parameters
    ----------
    df : pd.DataFrame
        Candidates DataFrame returned by _banks_to_candidates or similar
    bank : str
        Bank name
    category : str
        Category name
    amount : float
        Amount to add to limit_used
    
    Returns
    -------
    pd.DataFrame
        Updated DataFrame with modified limit_used and status
    """
    d = df.copy()

    bank_in = str(bank).strip()
    cat_in = str(category).strip()

    if 'bank' not in d.columns or 'category' not in d.columns:
        raise ValueError("DataFrame must contain 'bank' and 'category' columns")

    d['bank'] = d['bank'].astype(str).str.strip()
    d['category'] = d['category'].astype(str).str.strip()

    mask = (d['bank'] == bank_in) & (d['category'] == cat_in)
    if not mask.any():
        raise ValueError(f"Category '{category}' for bank '{bank}' not found")

    # Ensure numeric monitoring columns
    if 'limit_used' not in d.columns:
        d['limit_used'] = 0.0
    d['limit_used'] = pd.to_numeric(d['limit_used'], errors='coerce').fillna(0.0).astype(float)

    if 'm_limit' not in d.columns:
        raise ValueError("DataFrame must contain 'm_limit' column")
    d['m_limit'] = pd.to_numeric(d['m_limit'], errors='coerce').fillna(0.0).astype(float)

    d.loc[mask, 'limit_used'] = d.loc[mask, 'limit_used'] + float(amount)

    exhausted_mask = mask & (d['limit_used'] >= d['m_limit'])
    # Initialize status if absent
    if 'status' not in d.columns:
        d['status'] = 1

    # Handle limit_type if provided
    limit_type = (
        d.loc[mask, 'limit_type'].iloc[0].strip().lower()
        if 'limit_type' in d.columns else 'category'
    )
    
    if exhausted_mask.any():
        if limit_type == 'all':
            # Disable all categories of this bank
            d.loc[d['bank'] == bank_in, 'status'] = 0
        else:
            # Disable only this category
            d.loc[exhausted_mask, 'status'] = 0

    return d


def write_best_categories_to_txt(
    df: pd.DataFrame,
    filepath: str,
    active_only: bool = False,
    encoding: str = 'utf-8'
) -> None:
    """
    Write the best categories table to a text file in a readable format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Result from select_best_global (must have rank, bank, category, ratio, m_limit, status, etc.)
    filepath : str
        Output file path (e.g. 'best_categories.txt')
    active_only : bool
        If True, write only rows with status != 0 (active categories)
    encoding : str
        File encoding (default utf-8 for Cyrillic)
    """
    out = df.copy()
    if active_only:
        out = out[out['status'] != 0].reset_index(drop=True)
    cols = [c for c in ['rank', 'bank', 'category', 'ratio', 'm_limit', 'limit_type', 'status', 'locked'] if c in out.columns]
    out = out[cols]
    with open(filepath, 'w', encoding=encoding) as f:
        f.write("=" * 70 + "\n")
        f.write("BEST CASHBACK CATEGORIES BY BANK (CONSTRAINED SELECTION)\n")
        f.write("Max total_cats per bank; categories can repeat across banks. MTS deprioritized.\n")
        f.write("locked=1: pre-selected by you (fixed). locked=0: chosen by algorithm.\n")
        f.write("=" * 70 + "\n\n")
        f.write(out.to_string(index=False))
        f.write("\n\n")
        f.write("Columns: rank, bank, category, ratio (cashback %), m_limit (monthly limit RUR), limit_type, status (1=active), locked (1=fixed by you)\n")
    return None


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(output_txt: str = "best_categories.txt"):
    """
    Select the best categories across all banks (MTS last — points, not real money)
    and write the result to a text file.
    """
    # Create best categories with MTS as low priority (pick MTS only as afterthought)
    print("Selecting best categories (MTS deprioritized — points, not real money)...")
    best_cats_month = select_best_global(
        banks_dict,
        low_priority_banks=['mts'],
        locked_selections=LOCKED_SELECTIONS,
    )

    # Write full table to txt
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, output_txt)
    write_best_categories_to_txt(best_cats_month, out_path, active_only=False, encoding='utf-8')
    print(f"Result written to: {out_path}")

    # Also write active-only summary
    active_path = os.path.join(base_dir, "best_categories_active.txt")
    write_best_categories_to_txt(best_cats_month, active_path, active_only=True, encoding='utf-8')
    print(f"Active categories only written to: {active_path}")

    print("\nTop 10 best categories (real money first, MTS last):")
    try:
        print(best_cats_month.head(10).to_string(index=False))
    except UnicodeEncodeError:
        print("(See", out_path, "for full table with Cyrillic)")
    return best_cats_month


if __name__ == "__main__":
    best_categories = main()
