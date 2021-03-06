# Generated by Django 3.1.4 on 2021-04-10 03:05

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ObjectiveCategory',
            fields=[
                ('id_objective_category', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Objectives',
            fields=[
                ('id_objetives', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Subcategory',
            fields=[
                ('id_subcategory', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=250)),
                ('objective_category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.objectivecategory')),
            ],
        ),
        migrations.CreateModel(
            name='Questions',
            fields=[
                ('id_question', models.AutoField(primary_key=True, serialize=False)),
                ('question', models.TextField()),
                ('subcategory', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.subcategory')),
            ],
        ),
        migrations.AddField(
            model_name='objectivecategory',
            name='objettives',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.objectives'),
        ),
    ]
