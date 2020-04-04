package zzz_statics;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class sig_test {
    public static void main(String []args) throws Exception {
    	File f = new File("");
    	FileInputStream fin = new FileInputStream(f);
    	byte []reader = new byte[(int)f.length()];
    	fin.read(reader);
    	String s[] = new String(reader).split("\n");
    	for(int i = 0; i < s.length; i++) {
    		if(i % 3 == 0) {
    			
    		}
    	}
    	
    }
}
