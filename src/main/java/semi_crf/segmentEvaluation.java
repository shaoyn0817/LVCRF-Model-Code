package semi_crf;


import java.util.ArrayList;

import org.statnlp.commons.types.Instance;

import semi_crf.*;
import semi_crf.Span;
import semi_crf.semiCRFInstance;

public class segmentEvaluation {
    
    public static double eval(Instance[] instances) {
    	int all[] = new int[Label.LABELS.size()];
    	int find[] = new int[Label.LABELS.size()];
    	int right[] = new int[Label.LABELS.size()];
    	for(int i = 0; i < instances.length; i++){
    		semiCRFInstance instance = (semiCRFInstance) instances[i];
    		ArrayList<Span> originPrediction = instance.getPrediction();
    		ArrayList<Span> originOutput = instance.getOutput();
    		ArrayList<Span> output = addStart(originOutput);
    		ArrayList<Span> prediction = addStart(originPrediction);

    		for(int m = 0; m < prediction.size(); m++){
    			int begin = prediction.get(m)._start;
    			int end = prediction.get(m)._end;
    			int label = prediction.get(m)._label._id;
    			find[label]++;
    			
    			for(int n = 0; n < output.size(); n++){
        			int begin2 = output.get(n)._start;
        			int end2 = output.get(n)._end;
        			int label2 = output.get(n)._label._id;
        			if(begin == begin2 && end == end2 && label == label2){
        				right[label]++;
        				break;
        			}
    			}		
    		}
 		
    		for(int n = 0; n < output.size(); n++){
    			int label = output.get(n)._label._id;
    			all[label]++; 
    		}	
    	}
    	
    	int correct = 0;
    	int findall = 0;
    	int alllabel = 0;
    	for(int i = 0; i < semi_crf.Label.LABELS.size(); i++){
    		if(semi_crf.Label.get(i)._form.equals("O"))
    			continue;
    		correct += right[i];
    		findall += find[i];
    		alllabel += all[i];
    		double pre = (double)right[i]/find[i];
    		double rec = (double)right[i]/all[i]; 
    		System.out.println(semi_crf.Label.get(i)._form+":   Precision:"+pre
    				+"   Recall:"+rec+ "   F1:"+2*pre*rec/(pre+rec));
    	}

    	double pre = (double)correct/findall;
    	double rec = (double)correct/alllabel;
    	System.out.println("correct:"+correct+"   find:"+findall+"   alllabel:"+alllabel);
    	System.out.println("overall  Precision:" +pre+"   Recall:"+rec+"   F1:"+2*pre*rec/(pre+rec));
    	return 2*pre*rec/(pre+rec);
	} 
    
    public static ArrayList<Span> addStart(ArrayList<Span> output){
    	for(int i = 0; i < output.size(); i ++){
    		if(i == 0){
    			output.get(i)._start = 0;
    		} else {
    			output.get(i)._start = output.get(i-1)._end+1;
    		}
    	}
    	return output;
    }
    
}
