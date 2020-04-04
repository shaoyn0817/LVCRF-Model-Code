package org.statnlp.example.mention_hypergraphBILU_GENIA;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
 

public class Embedding {
    public static HashMap<String, ArrayList<Double>> dict = new HashMap<String, ArrayList<Double>>();
    public Embedding(String path) throws Exception{
    	File f = new File(path);
    	BufferedReader reader = new BufferedReader(new FileReader(f));
    	String line = "";
    	while((line = reader.readLine()) != null) {
    		String[] info = line.trim().split(" ");
    		ArrayList<Double> inf = new ArrayList<>();
    		for(int i = 1; i < info.length; i++) {
    			inf.add(Double.valueOf(info[i]));
    		}
    		String key = info[0];
    		dict.put(key, inf);
    	}
    	dict.put("<unk>", new ArrayList<>(50));
    	reader.close();
    }
    public static ArrayList<Double> getembed(String word) {
    	word = word.toLowerCase();
    	if(dict.containsKey(word)) {
    		return dict.get(word);
    	} else {
    		return dict.get("<unk>");
    	}
    }
}
