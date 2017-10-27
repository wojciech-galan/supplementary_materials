#! /usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import os
import cPickle as pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# contig length was established based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5210529/
main_page = 'https://img.jgi.doe.gov/cgi-bin/vr/main.cgi?section=Viral&page=viralScaffoldList&taxon_oid={}'
fname = 'genomeSet61246_11-oct-2017.csv'
dirname = os.path.join('..', 'datasets')
outfile = os.path.join('..', 'datasets', datetime.date.today().strftime("%Y-%m-%d")+'.dump')


def get_lengths(driver, address):
    driver.get(address)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "yui-pg0-0-page-report"))
    )
    elem = driver.find_element_by_class_name("yui-pg-current")
    while elem.text.strip() != 'Page 1 of 1':
        el = driver.find_element_by_id('yui-pg0-0-rpp')
        opt_all = el.find_element_by_xpath("option[contains(text(), 'All')]")
        opt_all.click()
        elem = driver.find_element_by_class_name("yui-pg-current")
    elems = driver.find_elements_by_xpath('//td[@class="yui-dt0-col-SequenceLengthbp yui-dt-col-SequenceLengthbp yui-dt-sortable yui-dt-resizeable"]/div/a')
    return [int(elem.text) for elem in elems]


with open(os.path.join(dirname, fname)) as fh:
    df = pd.read_csv(fh, sep='\t',  lineterminator='\n', names=None)
    taxon_oids = df['Taxon OID'].values
    contig_count = df['Viral Contig Count'].values

ret_list = []
try:
    driver = webdriver.Firefox()
    for i, oid in enumerate(taxon_oids):
        print i, oid
        res = get_lengths(driver, main_page.format(oid))
        assert len(res) == contig_count[i]
        ret_list.append(res)
finally:
    driver.close()

pickle.dump(ret_list, open(outfile, 'wb', 0))