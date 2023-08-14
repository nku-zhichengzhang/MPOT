function [value] = read_Ini_File(FilePath,key)
 %首先判断配置文件是否存在
 value = ' ' ;
 if(exist(FilePath,'file') ~= 2)
	 return;
 end
 %检查文件中有无key值，如果有则直接读取并返回，否则返回''
 fid = fopen(FilePath);
 while ~feof(fid)
 	tline = fgetl(fid);
 	if ~ischar(tline) || isempty(tline)
 	%跳过无效行
 		continue;
	end
	tline(find(isspace(tline))) = []; %删除行中的空格
	Index = strfind(tline, [key '=']);
	if ~isempty(Index)
		%如果找到该配置项，则读取对应的value值
		ParamName = strsplit(tline, '=');
		value = ParamName {2};
		break;
	end
 end
 fclose(fid); %关闭文件
end