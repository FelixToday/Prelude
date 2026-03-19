#!/bin/bash
# CW OW trafficsilver_rb_CW wtfpad_CW front_CW regulator_CW
# AWF DF TikTok VarCNN TF RF ARES TMWF CountMamba Prelude
for dataset in CW OW trafficsilver_rb_CW wtfpad_CW front_CW regulator_CW # OW
do
    for method in AWF DF TikTok VarCNN TF RF ARES TMWF CountMamba Prelude
    do
        checkpoint_path="../../checkpoints_test"
        file_base_dir="auto"
        note="baseline_same"
        maximum_load_time=80
        drop_extra_time="True"
        num_workers=16
        Model_name="EM1"
        TAM_type="ED1"
        test_flag="True"
        use_idx="False"
        optim="False"
        train_epochs=10
        overlap_ratio=0
        valid_name="valid"
        python main.py --dataset ${dataset} --config config/${method}.ini \
            --checkpoint_path $checkpoint_path --file_base_dir $file_base_dir \
            --note $note --maximum_load_time $maximum_load_time \
            --drop_extra_time $drop_extra_time --num_workers $num_workers \
            --TAM_type $TAM_type --Model_name $Model_name --test_flag $test_flag \
            --train_epochs $train_epochs --optim $optim --use_idx $use_idx --overlap_ratio $overlap_ratio \
            --valid_name $valid_name

        for load_ratio in {10..30..100}
        do
            python test.py --dataset ${dataset} --config config/${method}.ini \
                --checkpoint_path $checkpoint_path --file_base_dir $file_base_dir \
                --note $note --maximum_load_time $maximum_load_time \
                --drop_extra_time $drop_extra_time --num_workers $num_workers \
                --TAM_type $TAM_type --Model_name $Model_name --test_flag $test_flag \
                --is_pr_auc False --load_ratio $load_ratio --overlap_ratio $overlap_ratio
        done
    done
done
