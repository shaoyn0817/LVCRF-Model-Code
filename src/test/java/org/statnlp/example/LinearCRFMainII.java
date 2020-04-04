package org.statnlp.example;

import static org.statnlp.commons.Utils.print;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import javax.sound.midi.MidiDevice.Info;

import org.statnlp.InitWeightOptimizerFactory;
import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.example.linear_crf_II.LinearCRFFeatureManager;
import org.statnlp.example.linear_crf_II.LinearCRFNetworkCompiler;
 
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.StringIndex;
import org.statnlp.hypergraph.NetworkConfig.ModelType;
import org.statnlp.hypergraph.NetworkConfig.StoppingCriteria;
import org.statnlp.util.GenericPipeline;

import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;

public class LinearCRFMainII {
	
	public static void main(String args[]) throws IOException, InterruptedException{
		runOldPipeline(args);
	}
	
	public static void evaluate(Instance[] instances){
		int countCorrect = 0;
		int countGold = 0;
		int count = 0;
		for(Instance instance: instances){
			LinearInstance<?> inst = (LinearInstance<?>)instance;
			countCorrect += inst.countNumCorrectlyPredicted();
			countGold += inst.size();
			if(inst.size() > 5 && count < 5){
				for(String[] input: inst.input){
					System.out.print(input[0]+" ");
				}
				System.out.println();
				System.out.println(inst.output);
				for(int i=0; i<10; i++){
					System.out.println(inst.getTopKPredictions().get(i));
				}
				count += 1;
			}
		}
		System.out.println("Correct/Gold: "+countCorrect+"/"+countGold);
		System.out.println(String.format("Accuracy: %.3f%%", 100.0*countCorrect/countGold));
	}
	
	public static void runOldPipeline(String[] args) throws IOException, InterruptedException{
		String trainPath = System.getProperty("trainPath", "data/con03/tt");
		String testPath = System.getProperty("testPath", "data/con03/tt");
		
		String resultPath = System.getProperty("resultPath", "lcrf.result");
		String modelPath = System.getProperty("modelPath", "lcrf.model");
		String logPath = System.getProperty("logPath", "lcrf.log");
		
		boolean writeModelText = Boolean.parseBoolean(System.getProperty("writeModelText", "false"));

		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = Boolean.parseBoolean(System.getProperty("generativeTraining", "false"));
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = Boolean.parseBoolean(System.getProperty("parallelTouch", "true"));
		NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY = Boolean.parseBoolean(System.getProperty("featuresFromLabeledOnly", "false"));
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = Boolean.parseBoolean(System.getProperty("cacheFeatures", "true"));
		NetworkConfig.L2_REGULARIZATION_CONSTANT = Double.parseDouble(System.getProperty("l2", "0.01")); //0.01
		NetworkConfig.NUM_THREADS = Integer.parseInt(System.getProperty("numThreads", "8"));
		
		NetworkConfig.MODEL_TYPE = ModelType.valueOf(System.getProperty("modelType", "CRF")); // The model to be used: CRF, SSVM, or SOFTMAX_MARGIN
		NetworkConfig.USE_BATCH_TRAINING = Boolean.parseBoolean(System.getProperty("useBatchTraining", "false")); // To use or not to use mini-batches in gradient descent optimizer
		NetworkConfig.BATCH_SIZE = Integer.parseInt(System.getProperty("batchSize", "1000"));  // The mini-batch size (if USE_BATCH_SGD = true)
		NetworkConfig.MARGIN = Double.parseDouble(System.getProperty("svmMargin", "1.0"));
		
		NetworkConfig.FEATURE_INIT_WEIGHT = 0.0;
		NetworkConfig.USE_NEURAL_FEATURES = false;
		NetworkConfig.REGULARIZE_NEURAL_FEATURES = true;
		NetworkConfig.OPTIMIZE_NEURAL = false;
		NetworkConfig.AVOID_DUPLICATE_FEATURES = false;
		String weightInitFile = null;
		
		int numIterations = Integer.parseInt(System.getProperty("numIter", "3000"));
		
		
		
		PrintStream outstream = new PrintStream(logPath);
		
		OptimizerFactory optimizerFactory;
		if(NetworkConfig.MODEL_TYPE.USE_SOFTMAX && !(NetworkConfig.USE_NEURAL_FEATURES && !NetworkConfig.OPTIMIZE_NEURAL)){
			optimizerFactory = OptimizerFactory.getLBFGSFactory();
		} else {
			optimizerFactory = OptimizerFactory.getGradientDescentFactoryUsingAdaMThenStop();
		}

		GlobalNetworkParam param = new GlobalNetworkParam(optimizerFactory);

		int numTrain = -1;
		LinearInstance<Label>[] trainInstances = readCoNLLData(param, trainPath, true, true, numTrain);
		int size = trainInstances.length;
		System.err.println("Label size:"+LABELS.size()+".  Read.."+size+" instances from "+trainPath);
		
		LinearCRFNetworkCompiler compiler = new LinearCRFNetworkCompiler(LABELS.values());
		//LinearCRFFeatureManager fm = new LinearCRFFeatureManager(param, argsToFeatureManager);
		LinearCRFFeatureManager fm = new LinearCRFFeatureManager(param, LABELS_INDEX);
		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler, outstream);
		//model.visualize(LinearCRFViewer.class, trainInstances);
		
		model.train(trainInstances, numIterations);
		
		LinearInstance<Label>[] testInstances = readCoNLLData(param, testPath, true, false);
//		testInstances = Arrays.copyOf(testInstances, 1);
		int k = 8;
		Instance[] predictions = model.decode(testInstances, k);
		
		PrintStream[] outstreams = new PrintStream[]{outstream, System.out};
		PrintStream resultStream = new PrintStream(resultPath);
		
		int corr = 0;
		int total = 0;
		int count = 0;
		//evaluate the prediction results
		HashMap<String, Integer> all = new HashMap<String, Integer>();
		HashMap<String, Integer> find = new HashMap<String, Integer>();
		HashMap<String, Integer> right = new HashMap<String, Integer>();
		FileOutputStream ffout = new FileOutputStream("find.out");
		for(Instance ins: predictions){
			LinearInstance<Label> instance = (LinearInstance<Label>)ins;
			List<Label> goldLabel = instance.getOutput();
			List<Label> actualLabel = instance.getPrediction(); 
			ArrayList<String[]> words = (ArrayList<String[]>)instance.getInput();
			int aac = 0;
			for(int i=0; i<goldLabel.size(); i++){
				//f
				String glabel = goldLabel.get(i).getForm();
				if(glabel.indexOf("-") != -1) {
					glabel = glabel.substring(2);
				}
				String alabel = actualLabel.get(i).getForm();
				if(alabel.indexOf("-") != -1) {
					alabel = alabel.substring(2);
				}
				if(all.containsKey(glabel)) {
					int num = all.get(glabel)+1;
					all.put(glabel, num);
				} else all.put(glabel, 1);
				if(find.containsKey(alabel)) {
					int num = find.get(alabel)+1;
					find.put(alabel, num);
				} else find.put(alabel, 1);
				if(alabel.equals(glabel)) {
					if(right.containsKey(alabel)) {
						int num = right.get(alabel)+1;
						right.put(alabel, num);
					} else right.put(alabel, 1);
				}	
				if(!goldLabel.get(i).equals(actualLabel.get(i)))
					aac += 1;

				if(goldLabel.get(i).equals(actualLabel.get(i))){
					corr++;
				}
				total++;
				 
			}
			
			ffout.write((instance.toString()+"\n").getBytes());
			
			aac = 0;
			count++;
			 
		}
		resultStream.close();
		if(all.size() > find.size() || all.size() > right.size()) {
			for(String key : all.keySet()) {
				if(!find.containsKey(key)) find.put(key, 1);
				if(!right.containsKey(key)) right.put(key, 1);
			}
		}
		print(String.format("Correct/Total: %d/%d", corr, total), outstreams);
		print(String.format("Accuracy: %.2f%%", 100.0*corr/total), outstreams);
		int globalright = 0;
		int globalfind = 0;
		int globalall = 0;
		for(String key : all.keySet()) {
			if(key.equals("O"))
				continue;
			globalall += all.get(key);
			globalright += right.get(key);
			globalfind += find.get(key);
			double p = ((double)right.get(key))/(double)find.get(key);
			double r = (double)right.get(key)/(double)all.get(key);
			double f = 2*(p*r)/(p+r);
			System.out.println("Label("+key+")  p:"+p+"  r:"+r+"  f:"+f);
		}
		double p = (double)globalright/(double)globalfind;
		double r = (double)globalright/(double)globalall;
		double f = 2*(p*r)/(p+r);
		System.out.println("Global evaluation  p:"+p+"  r:"+r+"  f:"+f);
		outstream.close();
	}
	
	
	private static LinearInstance<Label>[] readCoNLLData(GlobalNetworkParam param, String fileName, boolean withLabels, boolean isLabeled, int number) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<LinearInstance<Label>> result = new ArrayList<LinearInstance<Label>>();
		ArrayList<String[]> words = null;
		ArrayList<Label> labels = null;
		int instanceId = 1;
		while(br.ready()){
			if(words == null){
				words = new ArrayList<String[]>();
			}
			if(withLabels && labels == null){
				labels = new ArrayList<Label>();
			}
			String line = br.readLine().trim();
			if(line.length() == 0){
				LinearInstance<Label> instance = new LinearInstance<Label>(instanceId, 1, words, labels);
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				if(result.size()==number) break;
				words = null;
				labels = null;
			} else {
				int lastSpace = line.lastIndexOf(" ");
				String[] features = line.substring(0, lastSpace).split(" ");
				words.add(features);
				if(withLabels){
					String info = line.substring(lastSpace+1);
					String infos[] = info.split("-");
					String tmp_format = "";
					String tag = "";
					if(infos.length == 1) {
						tmp_format = "O";
					    tag = "O";
					} else {
						tmp_format = infos[0];
						tag = infos[1];
					}
					Label label = getLabel(tag);
					label.setFormat(tmp_format);
					labels.add(label);
				}
			}
		}
		br.close();
		return result.toArray(new LinearInstance[result.size()]);
	}
	
	private static LinearInstance<Label>[]  readCoNLLData(GlobalNetworkParam param, String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		return readCoNLLData(param, fileName, withLabels, isLabeled, -1);
	}
	

	public static final Map<String, Label> LABELS = new HashMap<String, Label>();
	public static final Map<Integer, Label> LABELS_INDEX = new HashMap<Integer, Label>();
	
	public static Label getLabel(String form){
		if(!LABELS.containsKey(form)){
			Label label = new Label(form, LABELS.size());
			LABELS.put(form, label);
			LABELS_INDEX.put(label.getId(), label);
		}
		return LABELS.get(form);
	}
	
	public static Label getLabel(int id){
		return LABELS_INDEX.get(id);
	}
	
	public static void reset(){
		LABELS.clear();
		LABELS_INDEX.clear();
	}

	
}
