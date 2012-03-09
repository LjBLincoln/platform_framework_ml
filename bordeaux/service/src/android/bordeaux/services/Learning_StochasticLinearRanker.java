/*
 * Copyright (C) 2012 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package android.bordeaux.services;

import android.content.Context;
import android.bordeaux.learning.StochasticLinearRanker;
import android.util.Log;
import java.util.List;
import java.util.ArrayList;
import java.io.*;
import java.util.Scanner;

public class Learning_StochasticLinearRanker extends ILearning_StochasticLinearRanker.Stub {

    String TAG = "ILearning_StochasticLinearRanker";
    Context mContext;
    private StochasticLinearRanker mLearningSlRanker = null;

    public Learning_StochasticLinearRanker(Context context){
        mContext = context;
    }

    public boolean UpdateClassifier(List<StringFloat> sample_1, List<StringFloat> sample_2){
        ArrayList<StringFloat> temp_1 = (ArrayList<StringFloat>)sample_1;
        String[] keys_1 = new String[temp_1.size()];
        float[] values_1 = new float[temp_1.size()];
        for (int i = 0; i < temp_1.size(); i++){
            keys_1[i] = temp_1.get(i).key;
            values_1[i] = temp_1.get(i).value;
        }
        ArrayList<StringFloat> temp_2 = (ArrayList<StringFloat>)sample_2;
        String[] keys_2 = new String[temp_2.size()];
        float[] values_2 = new float[temp_2.size()];
        for (int i = 0; i < temp_2.size(); i++){
            keys_2[i] = temp_2.get(i).key;
            values_2[i] = temp_2.get(i).value;
        }
        if (mLearningSlRanker == null) mLearningSlRanker = new StochasticLinearRanker();
        boolean res = mLearningSlRanker.updateClassifier(keys_1,values_1,keys_2,values_2);
        return res;
    }

    public float ScoreSample(List<StringFloat> sample) {
        ArrayList<StringFloat> temp = (ArrayList<StringFloat>)sample;
        String[] keys = new String[temp.size()];
        float[] values = new float[temp.size()];
        for (int i = 0; i < temp.size(); i++){
            keys[i] = temp.get(i).key;
            values[i] = temp.get(i).value;
        }
        if (mLearningSlRanker == null) mLearningSlRanker = new StochasticLinearRanker();
        float res=mLearningSlRanker .scoreSample(keys,values);
        res = (float) (Math.exp(res)/(Math.exp(res)+Math.exp(-res)));
        return res;
    }

    public void LoadModel(String FileName){
        try{
            String str = "";
            StringBuffer buf = new StringBuffer();
            FileInputStream fis = mContext.openFileInput(FileName);
            BufferedReader reader = new BufferedReader(new InputStreamReader(fis));
            if (fis!=null) {
                while ((str = reader.readLine()) != null) {
                    buf.append(str + "\n" );
                }
            }
            fis.close();
            String Temps = buf.toString();
            String[] TempS_Array;
            TempS_Array = Temps.split("<>");
            String KeyValueString = TempS_Array[0];
            String ParamString = TempS_Array[1];
            String[] TempS1_Array;
            TempS1_Array = KeyValueString.split("\\|");
            int len = TempS1_Array.length;
            String[] keys = new String[len];
            float[] values = new float[len];
            for (int i =0; i< len; i++ ){
                String[] TempSd_Array;
                TempSd_Array = TempS1_Array[i].split(",");
                keys[i] = TempSd_Array[0];
                values[i] = Float.valueOf(TempSd_Array[1].trim()).floatValue();
            }
            String[] TempS2_Array;
            TempS2_Array = ParamString.split("\\|");
            int lenParam = TempS2_Array.length - 1;
            float[] parameters = new float[lenParam];
            for (int i =0; i< lenParam; i++ ){
                parameters[i] = Float.valueOf(TempS2_Array[i].trim()).floatValue();
            }
            if (mLearningSlRanker == null) mLearningSlRanker = new StochasticLinearRanker();
            boolean res = mLearningSlRanker.loadModel(keys,values, parameters);

        } catch (IOException e){
        }
    }

    public String SaveModel(String FileName){
        ArrayList<String> keys_list = new ArrayList<String>();
        ArrayList<Float> values_list = new ArrayList<Float>();
        ArrayList<Float> parameters_list = new ArrayList<Float>();
        if (mLearningSlRanker == null) mLearningSlRanker = new StochasticLinearRanker();
        mLearningSlRanker.getModel(keys_list,values_list, parameters_list);
        String S_model = "";
        for (int i = 0; i < keys_list.size(); i++)
            S_model = S_model + keys_list.get(i) + "," + values_list.get(i) + "|";
        String S_param ="";
        for (int i=0; i< parameters_list.size(); i++)
            S_param = S_param + parameters_list.get(i) + "|";
        String Final_Str = S_model + "<> " + S_param;
        try{
            FileOutputStream fos = mContext.openFileOutput(FileName, Context.MODE_PRIVATE);
            fos.write(Final_Str.getBytes());
            fos.close();
        } catch (IOException e){
        }
        return S_model;
    }
}
