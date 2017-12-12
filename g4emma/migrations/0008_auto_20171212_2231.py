# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-12-12 22:31
from __future__ import unicode_literals

import datetime
from django.db import migrations, models
import django.utils.timezone
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('g4emma', '0007_auto_20171212_2209'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='last_mod_date',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name='document',
            name='upload_date',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='document',
            name='filename',
            field=models.CharField(default=datetime.datetime(2017, 12, 12, 22, 31, 42, 642055, tzinfo=utc), max_length=255),
        ),
    ]
