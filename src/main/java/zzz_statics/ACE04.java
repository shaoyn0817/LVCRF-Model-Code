package zzz_statics;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.HashMap;

import scala.math.Integral;


public class ACE04 {
public static void main(String []args) throws Exception {
	HashMap<String, Integer> dict = new HashMap<String, Integer>();
	File f = new File("/home/shaoyn/shaoyn/dataset/Conll2003data/eng.train");
	FileInputStream fin = new FileInputStream(f);
	byte reader[] = new byte[(int)f.length()];
	fin.read(reader);
	String s[] = new String(reader).split("\n");
	String label = "";
	Integer len = 0;
	Integer strlen = 0;
	Integer strlen2 = 0;
	Integer strlen3 = 0;
	Integer strlen4 = 0;
	Integer temps = 0;
	for(int i = 0; i < s.length; i++) {
		String seg[] = s[i].split(" ");
		if(seg.length < 4)
			{
			if(temps >= 60) {
				System.out.println(temps+"****");
			}
			if(temps > strlen) {
				strlen = temps;
				System.out.println(strlen);
				temps = 0;
				

			} else {
				temps = 0;
			}
			continue;
			}
		else if(seg[0].equals("-DOCSTART-"))
			continue;
		
		temps++;
		
		if(seg[3].startsWith("B-")) {
			label = seg[3].substring(2);
			len ++;
			continue;
		} else if(seg[3].startsWith("I-")) {
			String temp = seg[3].substring(2);
			if(!temp.equals(label)) {
				System.out.println("wrong bio format");
			}
			len ++;
			continue;
		} else if(seg[3].equals("O")) {
			if(!label.equals("")) {
				if(!dict.containsKey(label)) {
					dict.put(label, len);
					label = "";
					len = 0;
					continue;
				}
			    Integer or = dict.get(label);
			    if(or < len) {
			    	dict.put(label, len);
			    }
			    len = 0;
			}
			label = "";
		}
	}
	System.out.println(dict);
	System.out.println(strlen);
	fin.close();
	
}
}
