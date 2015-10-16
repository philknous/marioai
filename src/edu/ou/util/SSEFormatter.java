package edu.ou.util;

import java.util.logging.Formatter;
import java.util.logging.LogRecord;

public class SSEFormatter extends Formatter {

	@Override
	public String format(LogRecord record) {
		StringBuilder sb = new StringBuilder();
		sb.append(record.getMessage());
		
		return sb.toString();
	}

}
