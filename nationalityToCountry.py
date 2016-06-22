import re


import pycountry

countries = [country.name for country in pycountry.countries]


def convert(word):
    patterns = ['ese', 'ian', 'an', 'ean', 'n', 'ic', 'ern']

    suffixes = ['a', 'o']

    for pattern in patterns:
        tup = re.findall(r'^(.*)(' + pattern + ')', word)

        if tup:
            country = tup[0][0]
            if country in countries:
                return country
            else:
                for suffix in suffixes:
                    new_country = country + suffix
                    if new_country in countries:
                        return new_country
