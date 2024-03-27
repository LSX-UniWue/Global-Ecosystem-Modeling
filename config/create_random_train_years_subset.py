import random


def select_random_years(years, percentage, seed=None):
    random.seed(seed)
    num_years_to_select = int(len(years) * percentage / 100)
    selected_years = random.sample(years, num_years_to_select)

    return sorted(selected_years)


# Example usage with seed
years_list = [1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
              1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]

print(len(years_list))

seed_value = 1337
for percentage in (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90):
    print(f"Selected Random Years ({percentage}%):", select_random_years(years_list, percentage, seed=seed_value))
